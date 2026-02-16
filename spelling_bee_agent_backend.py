# SPDX-FileCopyrightText: Copyright (c) 2026 Your Name or Organization
# SPDX-License-Identifier: BSD 2-Clause License

"""
Spelling Bee Voice Agent Backend
- Extracts word list from uploaded image using OCR
- Uses Langchain with Redis for agentic memory
- Integrates NeMo Guardrails for topic restriction
- Provides FastAPI endpoints for image upload, WebSocket, and session management
"""

import os
import base64
import json
import re
from pathlib import Path
from urllib import request, error
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import redis
import uvicorn

try:
    from langchain_community.chat_message_histories import RedisChatMessageHistory
except ImportError:
    RedisChatMessageHistory = None

try:
    from nemoguardrails import LLMRails, RailsConfig
except ImportError:
    LLMRails = None
    RailsConfig = None

try:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import Frame, LLMMessagesFrame, OutputAudioRawFrame, OutputTransportMessageFrame, TextFrame, TTSAudioRawFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.serializers.protobuf import ProtobufFrameSerializer

    from nvidia_pipecat.pipeline.ace_pipeline_runner import ACEPipelineRunner, PipelineMetadata
    from nvidia_pipecat.processors.nvidia_context_aggregator import create_nvidia_context_aggregator
    from nvidia_pipecat.processors.transcript_synchronization import (
        BotTranscriptSynchronization,
        UserTranscriptSynchronization,
    )
    from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
    from nvidia_pipecat.transports.network.ace_fastapi_websocket import ACETransport, ACETransportParams
    from nvidia_pipecat.transports.services.ace_controller.routers.websocket_router import (
        router as websocket_router,
    )

    # Cloud-hosted ASR + TTS (ElevenLabs)
    from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    # Patch: protobuf serializer uses exact type matching, so TTSAudioRawFrame
    # (subclass of OutputAudioRawFrame) gets silently dropped. Register it.
    ProtobufFrameSerializer.SERIALIZABLE_TYPES[TTSAudioRawFrame] = "audio"

    PIPECAT_AVAILABLE = True
except ImportError:
    PIPECAT_AVAILABLE = False

# Placeholder for NeMo Guardrails integration
# from nemoguardrails import Rails

import logging

logging.getLogger("pipecat.serializers.protobuf").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path("ui")
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# Redis setup for Langchain memory
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL)

# Session state (for demo, use Redis in production)
session_wordlists = {}  # session_id: [words]
session_progress = {}   # session_id: {current, incorrect, skipped}

VLLM_VL_BASE = os.getenv("VLLM_VL_BASE", "http://vllm-nemotron-nano-vl-8b:5566/v1")
VLLM_VL_MODEL = os.getenv("VLLM_VL_MODEL", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8")
NEMO_GUARDRAILS_CONFIG_PATH = os.getenv("NEMO_GUARDRAILS_CONFIG_PATH", "./guardrails")
ENABLE_NEMO_GUARDRAILS = os.getenv("ENABLE_NEMO_GUARDRAILS", "true").lower() == "true"

ACE_RUNNER = None
NEMO_RAILS = None
NEMO_POLICY_TEXT = ""


class _NoOpMemory:
    def save_context(self, _input, _output):
        return None


if PIPECAT_AVAILABLE:
    class MarkdownStripper(FrameProcessor):
        """Strip markdown from text frames and emit bot text as MessageFrame.

        The TTS consumes TextFrames so the client never sees bot text.
        This processor emits a serializable MessageFrame with tts_update
        JSON so the client can display transcripts and track score.
        """

        _MARKDOWN_RE = re.compile(r'[*_~`#]+')

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            if isinstance(frame, TextFrame) and frame.text:
                cleaned = self._MARKDOWN_RE.sub('', frame.text)
                frame.text = cleaned
                # Send cleaned text to client via OutputTransportMessageFrame
                msg = json.dumps({"type": "tts_update", "text": cleaned})
                await self.push_frame(OutputTransportMessageFrame(message=msg), direction)
            await super().process_frame(frame, direction)


def get_session_words(session_id: str) -> List[str]:
    if session_id in session_wordlists:
        return session_wordlists[session_id]

    raw = redis_client.get(f"spellingbee:session:{session_id}:words")
    if not raw:
        return []
    try:
        decoded = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        return decoded if isinstance(decoded, list) else []
    except json.JSONDecodeError:
        return []


def set_session_words(session_id: str, words: List[str]) -> None:
    session_wordlists[session_id] = words
    redis_client.set(f"spellingbee:session:{session_id}:words", json.dumps(words), ex=60 * 60 * 24)


def _load_guardrails_policy_text() -> str:
    rails_file = Path(NEMO_GUARDRAILS_CONFIG_PATH) / "rails.co"
    if rails_file.exists():
        return rails_file.read_text(encoding="utf-8").strip()
    return ""


def initialize_nemo_guardrails() -> None:
    global NEMO_RAILS, NEMO_POLICY_TEXT
    NEMO_POLICY_TEXT = _load_guardrails_policy_text()

    if not ENABLE_NEMO_GUARDRAILS or not LLMRails or not RailsConfig:
        return

    config_path = Path(NEMO_GUARDRAILS_CONFIG_PATH)
    if not config_path.exists():
        return

    try:
        # Read config and resolve env-var placeholders that NeMo doesn't expand
        raw_yaml = (config_path / "config.yml").read_text()
        resolved_model = os.getenv("NVIDIA_LLM_MODEL", VLLM_VL_MODEL)
        raw_yaml = raw_yaml.replace("__NVIDIA_LLM_MODEL__", resolved_model)
        rails_config = RailsConfig.from_content(yaml_content=raw_yaml, config_path=str(config_path))
        NEMO_RAILS = LLMRails(rails_config)
    except Exception as e:
        logger.warning("NeMo Guardrails init failed: %s", e)
        NEMO_RAILS = None


def _post_vllm_chat(messages: list[dict], max_tokens: int = 256, temperature: float = 0.2) -> str:
    payload = {
        "model": VLLM_VL_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{VLLM_VL_BASE}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=45) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed["choices"][0]["message"]["content"].strip()


def _parse_word_list(text: str) -> List[str]:
    cleaned = []
    for line in text.splitlines():
        token = re.sub(r"[^A-Za-z-]", "", line.strip())
        if token:
            cleaned.append(token)
    return cleaned

# --- OCR/vision-language model for extracting words from image ---
def extract_words_from_image(image_bytes) -> List[str]:
    image_bytes.seek(0)
    raw = image_bytes.read()
    if not raw:
        return []

    image_b64 = base64.b64encode(raw).decode("utf-8")
    vl_messages = [
        {
            "role": "system",
            "content": (
                "Extract only spelling words from the image. "
                "Return plain text with one single word per line and no numbering."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read this image and return only the spelling words."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        },
    ]

    try:
        vl_text = _post_vllm_chat(vl_messages, max_tokens=600, temperature=0.0)
        words = _parse_word_list(vl_text)
        if words:
            return words
    except (error.URLError, error.HTTPError, KeyError, IndexError, TimeoutError, json.JSONDecodeError):
        pass

    image_bytes.seek(0)
    image = Image.open(image_bytes)
    ocr_text = pytesseract.image_to_string(image)
    return [w.strip() for w in ocr_text.splitlines() if w.strip().isalpha()]


def generate_sentence_for_word(word: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are helping a child with spelling bee practice.",
        },
        {
            "role": "user",
            "content": (
                f"Use the word '{word}' in one short child-friendly sentence. "
                "Return only the sentence."
            ),
        },
    ]
    try:
        return _post_vllm_chat(messages, max_tokens=80, temperature=0.3)
    except (error.URLError, error.HTTPError, KeyError, IndexError, TimeoutError, json.JSONDecodeError):
        return f"Example: I can spell the word {word} today."


def generate_definition_for_word(word: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are helping a child with spelling bee practice.",
        },
        {
            "role": "user",
            "content": (
                f"Give a short child-friendly definition for the word '{word}'. "
                "Return one concise sentence."
            ),
        },
    ]
    try:
        return _post_vllm_chat(messages, max_tokens=100, temperature=0.2)
    except (error.URLError, error.HTTPError, KeyError, IndexError, TimeoutError, json.JSONDecodeError):
        return f"{word} is a spelling bee practice word."


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Image file is required.")

    session_id = os.urandom(8).hex()
    words = extract_words_from_image(file.file)

    if not words:
        raise HTTPException(status_code=422, detail="No spelling words detected in image.")

    set_session_words(session_id, words)
    session_progress[session_id] = {"current": 0, "incorrect": [], "skipped": []}

    return {
        "session_id": session_id,
        "word_count": len(words),
        "sample": words[:10],
        "next": "Connect websocket at /pipecat/ws?session_id=<session_id>",
    }


import random

SAMPLE_WORD_BANK = [
    "adventure", "beautiful", "calendar", "dangerous", "elephant",
    "favorite", "generous", "hurricane", "important", "knowledge",
    "language", "mountain", "necessary", "opposite", "paragraph",
    "question", "remember", "sentence", "together", "umbrella",
    "vacation", "wonderful", "alphabet", "birthday", "children",
    "discover", "exercise", "February", "grateful", "hospital",
]


@app.post("/sample-words")
async def sample_words(count: int = 5):
    """Generate a session with random sample words for quick testing."""
    words = random.sample(SAMPLE_WORD_BANK, min(count, len(SAMPLE_WORD_BANK)))
    session_id = os.urandom(8).hex()
    set_session_words(session_id, words)
    session_progress[session_id] = {"current": 0, "incorrect": [], "skipped": []}
    return {
        "session_id": session_id,
        "word_count": len(words),
        "sample": words,
        "next": f"Connect websocket at /pipecat/ws/{session_id}",
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "pipecat_available": PIPECAT_AVAILABLE}


@app.get("/")
async def home():
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    raise HTTPException(status_code=404, detail="UI not found")


if PIPECAT_AVAILABLE:
    async def create_pipecat_pipeline_task(pipeline_metadata: "PipelineMetadata"):
        session_id = pipeline_metadata.stream_id
        session_words = get_session_words(session_id) if session_id else []
        word_count = len(session_words)
        first_word = session_words[0] if session_words else None

        transport = ACETransport(
            websocket=pipeline_metadata.websocket,
            params=ACETransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        confidence=0.7,
                        min_volume=0.3,
                        start_secs=0.3,
                        stop_secs=1.2,
                    )
                ),
                audio_out_10ms_chunks=20,
            ),
        )

        llm = NvidiaLLMService(
            api_key="not-needed",
            base_url=os.getenv("NVIDIA_LLM_URL", VLLM_VL_BASE),
            model=os.getenv("NVIDIA_LLM_MODEL", VLLM_VL_MODEL),
        )

        stt = ElevenLabsRealtimeSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            sample_rate=16000,
        )

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_TTS_VOICE_ID", "9BWtsMINqrJLrRacOk9x"),
            model="eleven_turbo_v2_5",
            sample_rate=16000,
        )

        stt_transcript_synchronization = UserTranscriptSynchronization()
        tts_transcript_synchronization = BotTranscriptSynchronization()

        messages = [
            {
                "role": "system",
                "content": (
                    "You run a spelling bee. No markdown. Plain text only. Be EXTREMELY brief.\n\n"
                    "FLOW:\n"
                    "1. Say ONLY: 'Spell [word]. Say one letter at a time.' Then STOP.\n"
                    "2. Wait for the child to spell it.\n"
                    "3. If correct, say ONLY: 'Correct! Next word.' Then give the next word as in step 1.\n"
                    "4. If wrong, say ONLY: 'Not quite. The correct spelling is [word]. Next word.' Then give the next word.\n"
                    "5. If they say 'repeat', say ONLY the current word again.\n"
                    "6. If they say 'skip', say ONLY: 'Okay, next word.' Then give the next word.\n"
                    "7. When done, say: 'All done! You got [N] out of [total] correct.'\n\n"
                    "NEVER say more than one short sentence. NEVER explain, elaborate, encourage at length, or give definitions unless asked. "
                    "NEVER list multiple words. Only discuss spelling.\n\n"
                    f"Total words: {word_count}.\n"
                    + (
                        "WORD LIST (quiz in this order, never read aloud):\n"
                        + ", ".join(session_words)
                        + "\n"
                        if session_words else "No words uploaded yet.\n"
                    )
                    + f"{NEMO_POLICY_TEXT if NEMO_POLICY_TEXT else ''}"
                ),
            }
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = create_nvidia_context_aggregator(context, send_interims=False)

        md_stripper = MarkdownStripper()

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                stt_transcript_synchronization,
                context_aggregator.user(),
                llm,
                md_stripper,
                tts,
                tts_transcript_synchronization,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                send_initial_empty_metrics=True,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                start_metadata={"stream_id": pipeline_metadata.stream_id},
            ),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            del transport, client
            if session_words:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"Say exactly: 'Let us begin. Spell {first_word}. Say one letter at a time.' Nothing else."
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Say exactly: 'Please upload your spelling words first, then reconnect.' Nothing else."
                        ),
                    }
                )
            await task.queue_frames([LLMMessagesFrame(messages)])

        return task


if PIPECAT_AVAILABLE:
    initialize_nemo_guardrails()
    app.include_router(websocket_router, prefix="/pipecat")
    ACE_RUNNER = ACEPipelineRunner.create_instance(pipeline_callback=create_pipecat_pipeline_task)


if __name__ == "__main__":
    uvicorn.run("spelling_bee_agent_backend:app", host="0.0.0.0", port=8080, reload=False)
