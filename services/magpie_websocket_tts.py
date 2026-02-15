# Adapted from pipecat-ai/nemotron-january-2026 (MIT License)
#
# WebSocket TTS service for local Magpie TTS server.
#
# Uses WebSocket for token-by-token streaming from LLM to TTS:
# - Text tokens are sent immediately as they arrive from LLM
# - Audio streams back asynchronously on the same connection
# - Server accumulates text and flushes when enough audio is buffered

"""Magpie WebSocket TTS service for Pipecat pipeline."""

import asyncio
import json
import re
import time
from collections import deque
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import WebsocketTTSService

from services.frames import ChunkedLLMContinueGenerationFrame

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Install websockets: pip install websockets")
    raise

# Magpie outputs at 22kHz
MAGPIE_SAMPLE_RATE = 22000

# Regex pattern for emoji characters
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)

# Sentence boundary pattern
SENTENCE_BOUNDARY_PATTERN = re.compile(r'([.!?]["\'\)]*\s)')


def sanitize_text_for_tts(text: str) -> str:
    """Remove emojis and normalize special characters for TTS."""
    text = EMOJI_PATTERN.sub("", text)
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201C", '"')
    text = text.replace("\u201D", '"')
    text = text.replace("\u2014", "-")
    text = text.replace("\u2013", "-")
    return text


def split_into_sentences(text: str) -> list[str]:
    """Split text into individual sentences at boundary markers."""
    if not text:
        return []
    parts = SENTENCE_BOUNDARY_PATTERN.split(text)
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and SENTENCE_BOUNDARY_PATTERN.match(parts[i + 1]):
            sentences.append(parts[i] + parts[i + 1])
            i += 2
        elif parts[i]:
            sentences.append(parts[i])
            i += 1
        else:
            i += 1
    return sentences if sentences else [text] if text else []


class MagpieWebSocketTTSService(WebsocketTTSService):
    """WebSocket TTS service for local Magpie TTS server.

    Message protocol:
    - Send: {"type": "init", "voice": "...", "language": "...", "default_mode": "batch"}
    - Send: {"type": "text", "text": "...", "mode": "stream|batch", "preset": "..."}
    - Send: {"type": "close"} - triggers server to flush remaining text
    - Send: {"type": "cancel"} - abort immediately
    - Receive: binary audio data (16-bit PCM, 22kHz, mono)
    - Receive: {"type": "stream_created", ...}
    - Receive: {"type": "segment_complete", ...}
    - Receive: {"type": "done", ...}
    """

    class InputParams(BaseModel):
        language: str = "en"
        streaming_preset: str = "conservative"
        use_adaptive_mode: bool = True
        sentence_pause_ms: int = 250

    def __init__(
        self,
        *,
        server_url: str = "http://localhost:8001",
        voice: str = "aria",
        language: str = "en",
        sample_rate: Optional[int] = None,
        params: Optional["MagpieWebSocketTTSService.InputParams"] = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate or MAGPIE_SAMPLE_RATE,
            aggregate_sentences=False,
            **kwargs,
        )

        self._params = params or MagpieWebSocketTTSService.InputParams()

        # Convert http:// to ws://
        if server_url.startswith("http://"):
            server_url = "ws://" + server_url[7:]
        elif server_url.startswith("https://"):
            server_url = "wss://" + server_url[8:]

        self._server_url = server_url.rstrip("/")
        self._voice = voice.lower()
        self._language = language.lower()

        # Stream state
        self._stream_active = False
        self._stream_start_time: Optional[float] = None
        self._first_audio_received = False
        self._is_first_segment = True
        self._segment_sentence_boundary_queue: deque[bool] = deque()

        # Generation tracking for interruption handling
        self._gen = 0
        self._confirmed_gen = 0

        self._receive_task = None

        self.set_model_name("magpie-websocket")
        self.set_voice(voice)

        logger.info(
            f"MagpieWebSocketTTS initialized: server={self._server_url}, "
            f"voice={voice}, language={language}, "
            f"adaptive_mode={self._params.use_adaptive_mode}"
        )

    def can_generate_metrics(self) -> bool:
        return True

    def _ends_at_sentence_boundary(self, text: str) -> bool:
        text = text.strip()
        return bool(text) and text[-1] in ".!?"

    def _generate_silence_frames(self, duration_ms: int) -> bytes:
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return bytes(num_samples * 2)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption without disconnect/reconnect cycle."""
        await self.stop_all_metrics()
        self._gen += 1

        if self._websocket:
            try:
                await self._websocket.send(json.dumps({"type": "cancel"}))
            except Exception as e:
                logger.debug(f"Failed to send cancel: {e}")

        self._stream_active = False
        self._stream_start_time = None
        self._first_audio_received = False
        self._is_first_segment = True
        self._segment_sentence_boundary_queue.clear()

    async def _connect(self):
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

        self._stream_active = False
        self._stream_start_time = None
        self._first_audio_received = False
        self._is_first_segment = True
        self._segment_sentence_boundary_queue.clear()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            ws_url = f"{self._server_url}/ws/tts/stream"
            logger.debug(f"Connecting to WebSocket: {ws_url}")

            self._websocket = await websocket_connect(ws_url)

            await self._websocket.send(
                json.dumps({
                    "type": "init",
                    "voice": self._voice,
                    "language": self._language,
                })
            )

            await self._call_event_handler("on_connected")
            logger.info("Connected to Magpie TTS WebSocket")

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self.push_error(ErrorFrame(error=f"WebSocket connection failed: {e}"))
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Magpie TTS WebSocket")
                await self._websocket.close()
        except Exception as e:
            logger.debug(f"WebSocket close error: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                # Discard stale audio from previous generation
                if self._confirmed_gen != self._gen:
                    continue

                if not self._first_audio_received:
                    await self.stop_ttfb_metrics()
                    self._first_audio_received = True

                await self.push_frame(
                    TTSAudioRawFrame(message, self.sample_rate, 1)
                )

            elif isinstance(message, str):
                try:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "stream_created":
                        self._confirmed_gen = self._gen

                    elif msg_type == "segment_complete":
                        segment = msg.get("segment", 0)
                        audio_ms = msg.get("audio_ms", 0)
                        logger.debug(f"WS segment {segment} complete: {audio_ms:.0f}ms audio")

                        self._is_first_segment = False
                        self._first_audio_received = False

                        # Inject silence at sentence boundaries
                        if self._segment_sentence_boundary_queue:
                            ended_with_sentence = self._segment_sentence_boundary_queue.popleft()
                            if ended_with_sentence and self._params.sentence_pause_ms > 0:
                                silence = self._generate_silence_frames(
                                    self._params.sentence_pause_ms
                                )
                                await self.push_frame(
                                    TTSAudioRawFrame(silence, self.sample_rate, 1)
                                )

                        await self.push_frame(
                            ChunkedLLMContinueGenerationFrame(),
                            FrameDirection.UPSTREAM,
                        )

                    elif msg_type == "done":
                        total_ms = msg.get("total_audio_ms", 0)
                        segments = msg.get("segments_generated", 0)
                        logger.info(f"WS stream complete: {total_ms:.0f}ms audio, {segments} segments")
                        self._stream_active = False
                        self._is_first_segment = True
                        self._segment_sentence_boundary_queue.clear()
                        await self.push_frame(TTSStoppedFrame())

                    elif msg_type == "error":
                        error_msg = msg.get("message", "Unknown error")
                        is_fatal = msg.get("fatal", False)
                        logger.error(f"WS TTS error: {error_msg} (fatal={is_fatal})")
                        await self.push_frame(ErrorFrame(error=error_msg))
                        if is_fatal:
                            self._stream_active = False

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message[:100]}")

    async def flush_audio(self):
        """Send close message to trigger server to flush remaining text."""
        if self._websocket and self._stream_active:
            try:
                await self._websocket.send(json.dumps({"type": "close"}))
            except Exception as e:
                logger.debug(f"Failed to send close: {e}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Send text to TTS stream via WebSocket.

        Splits multi-sentence text into individual sentences to avoid GPU OOM on
        long chunks. In adaptive mode, the first segment uses streaming mode for
        fast TTFB, and subsequent segments use batch mode for higher quality.
        """
        await self.start_ttfb_metrics()

        text = sanitize_text_for_tts(text)
        if not text.strip():
            yield None
            return

        sentences = split_into_sentences(text)

        for sentence in sentences:
            if not sentence.strip():
                continue

            if not self._stream_active:
                self._stream_active = True
                self._stream_start_time = time.time()
                await self.push_frame(TTSStartedFrame())

            # Adaptive mode: stream first segment, batch thereafter
            if self._params.use_adaptive_mode and self._is_first_segment:
                mode = "stream"
                preset = self._params.streaming_preset
            else:
                mode = "batch"
                preset = None

            # Track sentence boundary for silence injection
            self._segment_sentence_boundary_queue.append(
                self._ends_at_sentence_boundary(sentence)
            )

            msg = {
                "type": "text",
                "text": sentence,
                "mode": mode,
            }
            if preset:
                msg["preset"] = preset

            try:
                if self._websocket:
                    await self._websocket.send(json.dumps(msg))
                    logger.debug(f"TTS sent ({mode}): {sentence[:40]}...")
            except Exception as e:
                logger.error(f"Failed to send text to TTS: {e}")

        yield None
