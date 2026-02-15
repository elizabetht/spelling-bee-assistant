# Adapted from pipecat-ai/nemotron-january-2026 (MIT License)
#
# NVIDIA Nemotron Speech WebSocket STT Service for Pipecat
#
# Connects to a local Nemotron Speech ASR server via WebSocket for streaming
# speech-to-text. The ASR server runs nvidia/nemotron-speech-streaming-en-0.6b.

"""NVIDIA Nemotron Speech streaming speech-to-text service implementation."""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional

import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601


class NVidiaWebSocketSTTService(WebsocketSTTService):
    """NVIDIA Nemotron Speech streaming speech-to-text service.

    Provides real-time speech recognition using NVIDIA's Nemotron Speech ASR model
    over WebSocket. Supports interim results for responsive transcription.

    The server expects:
    - Audio: 16-bit PCM, 16kHz, mono (raw bytes)
    - Reset signal: {"type": "reset", "finalize": true/false}

    The server sends:
    - Ready: {"type": "ready"}
    - Transcript: {"type": "transcript", "text": "...", "is_final": true/false, "finalize": true/false}
    """

    def __init__(
        self,
        *,
        url: str = "ws://localhost:8080",
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = url
        self._websocket = None
        self._receive_task: Optional[asyncio.Task] = None
        self._ready = False
        self._audio_send_lock = asyncio.Lock()
        self._audio_bytes_sent = 0

        # Frame ordering: hold UserStoppedSpeakingFrame until final transcript arrives
        self._waiting_for_final: bool = False
        self._pending_user_stopped_frame: Optional[UserStoppedSpeakingFrame] = None
        self._pending_frame_direction: FrameDirection = FrameDirection.DOWNSTREAM
        self._pending_frame_timeout_task: Optional[asyncio.Task] = None
        self._pending_frame_timeout_s: float = 0.5

        # STT processing time metric
        self._vad_stopped_time: Optional[float] = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._cancel_pending_frame_timeout()
        if self._pending_user_stopped_frame:
            await self.push_frame(
                self._pending_user_stopped_frame,
                self._pending_frame_direction,
            )
            self._pending_user_stopped_frame = None
        await self._send_reset(finalize=True)
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await self._cancel_pending_frame_timeout()
        self._pending_user_stopped_frame = None
        self._waiting_for_final = False
        await self._send_reset(finalize=True)
        if self._websocket and self._ready:
            try:
                msg = await asyncio.wait_for(self._websocket.recv(), timeout=0.5)
                data = json.loads(msg)
                if data.get("type") == "transcript" and data.get("is_final"):
                    await self._handle_transcript(data)
            except (asyncio.TimeoutError, Exception):
                pass
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if self._websocket and self._ready:
            try:
                async with self._audio_send_lock:
                    self._audio_bytes_sent += len(audio)
                    await self._websocket.send(audio)
            except Exception as e:
                logger.error(f"{self} failed to send audio: {e}")
                await self._report_error(ErrorFrame(f"Failed to send audio: {e}"))
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._cancel_pending_frame_timeout()
            self._pending_user_stopped_frame = None
            self._waiting_for_final = False
            self._vad_stopped_time = None
            await super().process_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            if self._waiting_for_final:
                self._pending_user_stopped_frame = frame
                self._pending_frame_direction = direction
                self._start_pending_frame_timeout()
                logger.debug(f"{self} holding UserStoppedSpeakingFrame")
                self._vad_stopped_time = time.time()
                await self._send_reset(finalize=True)
                return
            await super().process_frame(frame, direction)
            return

        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._waiting_for_final = True
            await self._send_reset(finalize=False)

    async def _send_reset(self, finalize: bool = True):
        if self._websocket and self._ready:
            try:
                async with self._audio_send_lock:
                    await self._websocket.send(json.dumps({
                        "type": "reset",
                        "finalize": finalize,
                    }))
                    samples = self._audio_bytes_sent // 2
                    duration_ms = (samples * 1000) // 16000
                    reset_type = "hard" if finalize else "soft"
                    logger.debug(f"{self} sent {reset_type} reset (audio: {duration_ms}ms)")
                    if finalize:
                        self._audio_bytes_sent = 0
            except Exception as e:
                logger.error(f"{self} failed to send reset: {e}")

    def _start_pending_frame_timeout(self):
        if self._pending_frame_timeout_task:
            self._pending_frame_timeout_task.cancel()
        self._pending_frame_timeout_task = asyncio.create_task(
            self._pending_frame_timeout_handler()
        )

    async def _pending_frame_timeout_handler(self):
        try:
            await asyncio.sleep(self._pending_frame_timeout_s)
            if self._pending_user_stopped_frame:
                logger.debug(f"{self} timeout waiting for final transcript, releasing frame")
                await self.push_frame(
                    self._pending_user_stopped_frame,
                    self._pending_frame_direction,
                )
                self._pending_user_stopped_frame = None
                self._waiting_for_final = False
        except asyncio.CancelledError:
            pass

    async def _cancel_pending_frame_timeout(self):
        if self._pending_frame_timeout_task:
            self._pending_frame_timeout_task.cancel()
            try:
                await self._pending_frame_timeout_task
            except asyncio.CancelledError:
                pass
            self._pending_frame_timeout_task = None

    async def _release_pending_frame(self):
        self._waiting_for_final = False
        if self._pending_user_stopped_frame:
            await self._cancel_pending_frame_timeout()
            logger.debug(f"{self} releasing UserStoppedSpeakingFrame")
            await self.push_frame(
                self._pending_user_stopped_frame,
                self._pending_frame_direction,
            )
            self._pending_user_stopped_frame = None

    async def _connect(self):
        logger.debug(f"{self} connecting to {self._url}")
        await self._connect_websocket()
        self._receive_task = asyncio.create_task(
            self._receive_task_handler(self._report_error)
        )
        await self._call_event_handler("on_connected", self)

    async def _disconnect(self):
        logger.debug(f"{self} disconnecting")
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        await self._disconnect_websocket()
        await self._call_event_handler("on_disconnected", self)

    async def _connect_websocket(self):
        try:
            self._websocket = await websockets.connect(self._url)
            self._ready = False
            try:
                ready_msg = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
                data = json.loads(ready_msg)
                if data.get("type") == "ready":
                    self._ready = True
                    logger.info(f"{self} connected and ready")
                else:
                    logger.warning(f"{self} unexpected initial message: {data}")
                    self._ready = True
            except asyncio.TimeoutError:
                logger.warning(f"{self} timeout waiting for ready, proceeding anyway")
                self._ready = True
        except Exception as e:
            logger.error(f"{self} connection failed: {e}")
            await self._report_error(ErrorFrame(f"Connection failed: {e}"))
            raise

    async def _disconnect_websocket(self):
        self._ready = False
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"{self} error closing websocket: {e}")
            finally:
                self._websocket = None

    async def _receive_messages(self):
        if not self._websocket:
            return
        async for message in self._websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "transcript":
                    await self._handle_transcript(data)
                elif msg_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    logger.error(f"{self} server error: {error_msg}")
                    await self._report_error(ErrorFrame(f"Server error: {error_msg}"))
                elif msg_type == "ready":
                    self._ready = True
                    logger.debug(f"{self} server ready")
                else:
                    logger.debug(f"{self} unknown message type: {msg_type}")
            except json.JSONDecodeError as e:
                logger.error(f"{self} invalid JSON: {e}")
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")

    async def _handle_transcript(self, data: dict):
        text = data.get("text", "")
        is_final = data.get("is_final", False)
        is_hard_reset = data.get("finalize", True)

        if not text:
            if is_final and is_hard_reset:
                await self._release_pending_frame()
            return

        await self.stop_ttfb_metrics()
        timestamp = time_now_iso8601()

        if is_final:
            reset_type = "hard" if is_hard_reset else "soft"
            logger.debug(f"{self} {reset_type} final: {text[-50:] if len(text) > 50 else text}")

            if is_hard_reset:
                if text:
                    await self.push_frame(
                        TranscriptionFrame(text, self._user_id, timestamp, language=None)
                    )
                    await self.stop_processing_metrics()

                    if self._vad_stopped_time is not None:
                        processing_time = time.time() - self._vad_stopped_time
                        logger.info(f"{self} NemotronSTT TTFB: {processing_time * 1000:.0f}ms")
                        metrics_frame = MetricsFrame(
                            data=[TTFBMetricsData(processor="NemotronSTT", value=processing_time)]
                        )
                        await self.push_frame(metrics_frame)
                        self._vad_stopped_time = None

                await self._release_pending_frame()
        else:
            logger.trace(f"{self} interim: {text[:30]}...")
            await self.push_frame(
                InterimTranscriptionFrame(text, self._user_id, timestamp, language=None)
            )

    async def start_metrics(self):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
