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
    from pipecat.frames.frames import (
        BotStoppedSpeakingFrame,
        Frame,
        InterimTranscriptionFrame,
        LLMMessagesFrame,
        OutputAudioRawFrame,
        OutputTransportMessageFrame,
        TextFrame,
        TranscriptionFrame,
        TTSAudioRawFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.serializers.protobuf import ProtobufFrameSerializer

    from nvidia_pipecat.pipeline.ace_pipeline_runner import ACEPipelineRunner, PipelineMetadata
    from nvidia_pipecat.frames.transcripts import BotUpdatedSpeakingTranscriptFrame
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
from loguru import logger as _loguru_logger

logging.getLogger("pipecat.serializers.protobuf").setLevel(logging.ERROR)
# Pipecat uses loguru; filter out noisy "not serializable" warnings
_loguru_logger.disable("pipecat.serializers.protobuf")
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

# NeMo Guardrails uses the OpenAI provider to talk to vLLM's OpenAI-compatible
# API.  Set OPENAI_API_KEY (dummy) and OPENAI_BASE_URL so the LangChain OpenAI
# client inside NeMo can connect to the self-hosted vLLM endpoint.
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "not-needed"
if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("NVIDIA_LLM_URL", VLLM_VL_BASE)

ACE_RUNNER = None
NEMO_RAILS = None
NEMO_POLICY_TEXT = ""





class _NoOpMemory:
    def save_context(self, _input, _output):
        return None


if PIPECAT_AVAILABLE:
    class MarkdownStripper(FrameProcessor):
        """Strip markdown characters from text frames before TTS."""

        _MARKDOWN_RE = re.compile(r'[*_~`#]+')

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TextFrame) and frame.text:
                frame.text = self._MARKDOWN_RE.sub('', frame.text)
            await self.push_frame(frame, direction)

    class UserTranscriptBridge(FrameProcessor):
        """Forward user STT transcription frames to the browser as JSON messages.

        The pipeline sends TranscriptionFrame/InterimTranscriptionFrame to the
        LLM context aggregator, but they never reach the transport output.
        This processor converts them to OutputTransportMessageFrame (asr_update /
        asr_end) so the browser can display what the child is saying.
        """

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, InterimTranscriptionFrame):
                msg = OutputTransportMessageFrame(
                    message={"type": "asr_update", "text": frame.text}
                )
                await self.push_frame(msg, direction)
            elif isinstance(frame, TranscriptionFrame):
                msg = OutputTransportMessageFrame(
                    message={"type": "asr_end", "text": frame.text}
                )
                await self.push_frame(msg, direction)
            await self.push_frame(frame, direction)

    class TranscriptBridge(FrameProcessor):
        """Convert nvidia-pipecat transcript frames to serializable transport messages.

        BotUpdatedSpeakingTranscriptFrame is a SystemFrame and cannot be
        serialized by ProtobufFrameSerializer.  This processor converts them
        to OutputTransportMessageFrame (tts_update / tts_end JSON) so the
        browser UI receives bot text for score parsing and transcript display.
        """

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, BotUpdatedSpeakingTranscriptFrame):
                msg = OutputTransportMessageFrame(
                    message={"type": "tts_update", "text": frame.transcript}
                )
                await self.push_frame(msg, direction)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                msg = OutputTransportMessageFrame(
                    message={"type": "tts_end"}
                )
                await self.push_frame(msg, direction)
            await self.push_frame(frame, direction)

    class GuardrailsFilter(FrameProcessor):
        """Pre-LLM processor that blocks off-topic user messages.

        Uses NeMo Guardrails (NEMO_RAILS) when available. Falls back to a
        keyword heuristic that detects common off-topic patterns and replaces
        the user transcription with a redirect message.
        """

        # Spelling-related keywords that indicate on-topic speech
        _ON_TOPIC_RE = re.compile(
            r'\b('
            r'spell|repeat|sentence|definition|meaning|skip|review|'
            r'start|continue|next|word|letter|practice|bee|done|'
            r'correct|wrong|yes|no|ready'
            r')\b',
            re.IGNORECASE,
        )
        # Single letters or letter sequences (the child is spelling)
        _LETTERS_RE = re.compile(
            r'^[A-Za-z](?:\s*[-\s]\s*[A-Za-z])*$'
        )

        _REDIRECT = "Let's get back to spelling practice."

        def __init__(self, messages: list):
            super().__init__()
            self._messages = messages

        def _is_on_topic(self, text: str) -> bool:
            """Return True if the text is related to spelling practice."""
            stripped = text.strip()
            if not stripped:
                return True
            # Single letters or letter sequences are always on-topic
            if self._LETTERS_RE.match(stripped):
                return True
            # Short utterances (≤3 words) are likely spelling attempts
            if len(stripped.split()) <= 3:
                return True
            # Check for spelling-related keywords
            if self._ON_TOPIC_RE.search(stripped):
                return True
            return False

        async def _check_guardrails_async(self, text: str) -> Optional[str]:
            """Run NeMo Guardrails if available. Returns redirect text or None."""
            if NEMO_RAILS is None:
                return None
            try:
                import asyncio
                result = await NEMO_RAILS.generate_async(
                    messages=[{"role": "user", "content": text}]
                )
                bot_msg = result.get("content", "") if isinstance(result, dict) else str(result)
                # NeMo returns a redirect message if off-topic
                if "spelling" in bot_msg.lower() or "back to" in bot_msg.lower():
                    return bot_msg
            except Exception as e:
                logger.warning("NeMo Guardrails check failed: %s", e)
            return None

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, TranscriptionFrame) and frame.text:
                text = frame.text.strip()

                # Fast path: check keyword heuristic first
                if not self._is_on_topic(text):
                    # Try NeMo Guardrails for a more nuanced check
                    redirect = await self._check_guardrails_async(text)
                    if redirect or not self._is_on_topic(text):
                        logger.info("GuardrailsFilter: blocked off-topic input: %r", text)
                        self._messages.append({
                            "role": "system",
                            "content": (
                                "[INTERNAL] The child said something off-topic. "
                                "Respond ONLY with: 'Let's get back to spelling "
                                "practice.' then re-announce the current word."
                            ),
                        })

            await self.push_frame(frame, direction)

    class ReviewInjector(FrameProcessor):
        """Pre-LLM processor that injects the review word list before the LLM responds.

        Sits before the LLM in the pipeline. When a user TranscriptionFrame
        arrives (child finished speaking), checks if review words should be
        injected so the LLM has the list available when generating its response.
        Triggers when the word count indicates all words have been quizzed,
        or when the user says "review".
        """

        _REVIEW_RE = re.compile(r'\breview\b', re.IGNORECASE)

        def __init__(self, session_id: str, word_count: int, messages: list):
            super().__init__()
            self._session_id = session_id
            self._word_count = word_count
            self._messages = messages
            self._injected = False
            self._words_judged = 0

        def notify_judgment(self):
            """Called by IncorrectWordTracker when a correct/incorrect judgment is detected."""
            self._words_judged += 1

        def _inject_if_needed(self, user_text: str = ""):
            if self._injected:
                return
            # Inject if all words have been judged or user asks for review
            should_inject = (
                (self._words_judged >= self._word_count and self._word_count > 0) or
                self._REVIEW_RE.search(user_text)
            )
            if not should_inject:
                return
            incorrect = get_session_incorrect_words(self._session_id)
            if not incorrect:
                return
            self._injected = True
            review_msg = {
                "role": "system",
                "content": (
                    f"REVIEW MODE: Quiz ONLY these {len(incorrect)} missed words: "
                    f"{', '.join(incorrect)}. "
                    f"Present ONE word at a time: 'Your word is [word].' then STOP and "
                    f"WAIT for the child to spell it. Do NOT judge until the child responds. "
                    f"Do NOT generate the child's answer. Do NOT auto-complete."
                ),
            }
            self._messages.append(review_msg)
            logger.info("ReviewInjector: injected review words into context: %s", incorrect)

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                self._inject_if_needed(frame.text or "")
            await self.push_frame(frame, direction)

    class SpellingVerifier(FrameProcessor):
        """Pre-LLM processor that verifies spelling server-side.

        - Strips ASR noise (parenthetical descriptions, dashes, trailing periods)
        - Parses letter sequences like "J-O-U-R-N-Y" into "JOURNY"
        - Handles ASR misrecognitions of letter names (e.g., "cue" → Q)
        - Compares to the current target word
        - Injects a SPELLING RESULT system message so the LLM can trust the verdict
        """

        # Matches (parenthetical noise) like "(child speaking indistinctly)"
        _NOISE_RE = re.compile(r'\([^)]*\)')
        # Matches letter sequences: "J-O-U-R-N-E-Y" or "J O U R N E Y"
        _LETTER_SEQ_RE = re.compile(
            r'\b([A-Za-z])(?:\s*[-\s]\s*([A-Za-z])){2,}\b'
        )
        # Broader pattern: split on dash/space and check if all tokens are single letters
        _DASH_SPLIT_RE = re.compile(r'[-\s]+')

        # Common ASR misrecognitions of spoken letter names
        _LETTER_NAME_MAP = {
            "ay": "A", "aye": "A", "eh": "A",
            "bee": "B", "be": "B",
            "see": "C", "sea": "C", "cee": "C",
            "dee": "D",
            "ee": "E",
            "ef": "F", "eff": "F",
            "gee": "G",
            "aitch": "H", "ache": "H", "each": "H",
            "eye": "I",
            "jay": "J",
            "kay": "K", "okay": "K",
            "el": "L", "elle": "L", "ell": "L",
            "em": "M",
            "en": "N",
            "oh": "O", "owe": "O",
            "pee": "P",
            "cue": "Q", "queue": "Q", "que": "Q",
            "are": "R", "ar": "R",
            "es": "S", "ess": "S",
            "tee": "T", "tea": "T",
            "you": "U", "ewe": "U",
            "vee": "V",
            "double you": "W", "double u": "W", "doubleyou": "W",
            "ex": "X",
            "why": "Y", "wie": "Y",
            "zee": "Z", "zed": "Z",
        }

        def __init__(self, session_id: str, session_words: List[str],
                     review_injector: "ReviewInjector", messages: list):
            super().__init__()
            self._session_id = session_id
            self._session_words = [w.lower() for w in session_words]
            self._review_injector = review_injector
            self._messages = messages
            self._review_index = 0

        def _clean_text(self, text: str) -> str:
            """Strip ASR noise from transcription text."""
            cleaned = self._NOISE_RE.sub('', text)
            # Remove leading dashes/bullets
            cleaned = re.sub(r'^[\s\-\u2022]+', '', cleaned)
            # Remove trailing periods that ASR adds
            cleaned = cleaned.rstrip('. ')
            return cleaned.strip()

        def _normalize_token(self, token: str) -> Optional[str]:
            """Normalize a single token to a letter, handling ASR misrecognitions."""
            t = token.strip()
            if len(t) == 1 and t.isalpha():
                return t.upper()
            mapped = self._LETTER_NAME_MAP.get(t.lower())
            return mapped if mapped else None

        def _extract_letters(self, text: str) -> Optional[str]:
            """Extract letter sequence from text like 'J-O-U-R-N-E-Y' or 'J O U R N E Y'.

            Also handles ASR misrecognitions where letters are transcribed as
            words (e.g., 'cue you eye ee tee el why' → 'QUIETLY').
            """
            cleaned = self._clean_text(text)
            if not cleaned:
                return None

            # Try splitting on dashes and/or spaces
            tokens = self._DASH_SPLIT_RE.split(cleaned)
            # Filter empty tokens
            tokens = [t for t in tokens if t]

            if len(tokens) < 2:
                return None

            # Try to normalize each token to a single letter
            letters = [self._normalize_token(t) for t in tokens]
            if all(l is not None for l in letters):
                return ''.join(letters)

            return None

        def _get_target_word(self) -> Optional[str]:
            """Get the current target word based on how many words have been judged."""
            if self._review_injector._injected:
                # Review phase: get words from Redis
                review_words = get_session_incorrect_words(self._session_id)
                if review_words and self._review_index < len(review_words):
                    return review_words[self._review_index].lower()
                return None

            idx = self._review_injector._words_judged
            if idx < len(self._session_words):
                return self._session_words[idx]
            return None

        def advance_review_index(self):
            """Called when a review word is judged."""
            self._review_index += 1

        _VERDICT_TAG = "[INTERNAL — DO NOT READ ALOUD OR RECAP]"

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, TranscriptionFrame) and frame.text:
                # Clean the transcription text for downstream processors
                cleaned = self._clean_text(frame.text)
                if cleaned != frame.text:
                    frame.text = cleaned

                # Only verify on FINAL transcriptions, not interims
                if isinstance(frame, InterimTranscriptionFrame):
                    await self.push_frame(frame, direction)
                    return

                # Try to extract a letter sequence and verify spelling
                letters = self._extract_letters(cleaned)
                if letters:
                    target = self._get_target_word()
                    if target:
                        spelled = letters.upper()
                        target_upper = target.upper()
                        verdict = "CORRECT" if spelled == target_upper else "INCORRECT"

                        # Determine the next word so the LLM doesn't have to
                        idx = self._review_injector._words_judged
                        next_idx = idx + 1
                        next_word = (
                            self._session_words[next_idx]
                            if next_idx < len(self._session_words)
                            else None
                        )

                        if verdict == "CORRECT":
                            if next_word:
                                say = f"Say EXACTLY: 'Correct! Your next word is {next_word}.'"
                            else:
                                say = "Say EXACTLY: 'Correct! All done!'"
                        else:
                            if next_word:
                                say = (
                                    f"Say EXACTLY: 'Not quite. The correct spelling is {target}. "
                                    f"Your next word is {next_word}.'"
                                )
                            else:
                                say = (
                                    f"Say EXACTLY: 'Not quite. The correct spelling is {target}. "
                                    f"All done!'"
                                )

                        # Remove any stale verdict messages before adding new one
                        self._messages[:] = [
                            m for m in self._messages
                            if not (m.get("role") == "system"
                                    and self._VERDICT_TAG in m.get("content", ""))
                        ]

                        result_msg = {
                            "role": "system",
                            "content": (
                                f"{self._VERDICT_TAG} "
                                f"Verdict: {verdict}. {say} "
                                f"Do NOT add anything else."
                            ),
                        }
                        self._messages.append(result_msg)
                        logger.info(
                            "SpellingVerifier: %s spelled='%s' target='%s' next='%s'",
                            verdict, spelled, target, next_word or 'LAST',
                        )

            await self.push_frame(frame, direction)

    class IncorrectWordTracker(FrameProcessor):
        """Track incorrect words in Redis by monitoring LLM text output.

        Watches TextFrames for "Not quite" (incorrect) responses and records
        the PREVIOUS word (the one being quizzed) to a Redis set.
        Also notifies ReviewInjector of each judgment so it knows when all
        words have been quizzed.
        """

        _NOT_QUITE_RE = re.compile(r'\bnot\s+quite\b', re.IGNORECASE)
        _CORRECT_RE = re.compile(r'\bcorrect\b', re.IGNORECASE)
        _WORD_RE = re.compile(
            r'(?:first|next|new|your)\s+(?:next\s+)?word\s+is[:\s]+["\']?(\w+)["\']?',
            re.IGNORECASE,
        )

        def __init__(self, session_id: str, session_words: List[str],
                     review_injector: Optional["ReviewInjector"] = None,
                     spelling_verifier: Optional["SpellingVerifier"] = None):
            super().__init__()
            self._session_id = session_id
            self._session_words = [w.lower() for w in session_words]
            self._redis_key = f"spellingbee:session:{session_id}:incorrect"
            self._current_word: Optional[str] = None
            self._buffer = ""
            self._marked_incorrect = False
            self._judged = False  # whether current word has been judged
            self._review_injector = review_injector
            self._spelling_verifier = spelling_verifier

        def _redis_add_incorrect(self, word: str):
            try:
                redis_client.sadd(self._redis_key, word.lower())
                redis_client.expire(self._redis_key, 60 * 60 * 24)
                logger.info("Incorrect word '%s' saved to Redis for session %s", word, self._session_id)
            except Exception as e:
                logger.warning("Failed to save incorrect word to Redis: %s", e)

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, TextFrame) and frame.text:
                self._buffer += frame.text

                # Detect incorrect judgment BEFORE extracting next word.
                # "Not quite" means the CURRENT word was wrong.
                if self._NOT_QUITE_RE.search(self._buffer) and self._current_word and not self._marked_incorrect:
                    self._redis_add_incorrect(self._current_word)
                    self._marked_incorrect = True
                    if not self._judged and self._review_injector:
                        self._judged = True
                        self._review_injector.notify_judgment()
                        if self._spelling_verifier and self._review_injector._injected:
                            self._spelling_verifier.advance_review_index()

                # Detect correct judgment
                if (self._CORRECT_RE.search(self._buffer) and
                        not self._NOT_QUITE_RE.search(self._buffer) and
                        not self._judged and self._current_word):
                    self._judged = True
                    if self._review_injector:
                        self._review_injector.notify_judgment()
                        if self._spelling_verifier and self._review_injector._injected:
                            self._spelling_verifier.advance_review_index()

                # Track which word is announced (updates current word for next round)
                m = self._WORD_RE.search(self._buffer)
                if m:
                    new_word = m.group(1).lower()
                    if new_word != self._current_word:
                        self._current_word = new_word
                        self._marked_incorrect = False
                        self._judged = False  # reset for new word

            elif isinstance(frame, BotStoppedSpeakingFrame):
                # Reset buffer between bot utterances
                self._buffer = ""

            await self.push_frame(frame, direction)


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


def get_session_incorrect_words(session_id: str) -> List[str]:
    """Retrieve the set of incorrect words for a session from Redis."""
    try:
        members = redis_client.smembers(f"spellingbee:session:{session_id}:incorrect")
        return [m.decode("utf-8") if isinstance(m, bytes) else m for m in members]
    except Exception:
        return []


def clear_session_incorrect_words(session_id: str) -> None:
    """Clear incorrect words after a review round."""
    try:
        redis_client.delete(f"spellingbee:session:{session_id}:incorrect")
    except Exception:
        pass


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
        # Load Colang files from the config directory
        colang_content = ""
        for co_file in sorted(config_path.glob("*.co")):
            colang_content += co_file.read_text() + "\n"
        rails_config = RailsConfig.from_content(
            yaml_content=raw_yaml,
            colang_content=colang_content,
        )
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
    "discover", "exercise", "february", "grateful", "hospital",
    "imagine", "journey", "kitchen", "library", "mystery",
    "notebook", "opinion", "pleasant", "quietly", "relative",
    "shoulder", "treasure", "uniform", "village", "weather",
    "yesterday", "absolute", "behavior", "confident", "decision",
]


@app.post("/sample-words")
async def sample_words(count: int = 25):
    """Generate a session with random sample words for quick testing."""
    words = random.sample(SAMPLE_WORD_BANK, min(count, len(SAMPLE_WORD_BANK)))
    session_id = os.urandom(8).hex()
    set_session_words(session_id, words)
    session_progress[session_id] = {"current": 0, "incorrect": [], "skipped": []}
    return {
        "session_id": session_id,
        "word_count": len(words),
        "sample": [],
        "next": f"Connect websocket at /pipecat/ws/{session_id}",
    }


@app.get("/session/{session_id}/incorrect")
async def get_incorrect(session_id: str):
    """Return the list of words the child got wrong during the session."""
    words = get_session_incorrect_words(session_id)
    return {"session_id": session_id, "incorrect_words": words, "count": len(words)}


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
        progress = session_progress.get(session_id, {})
        current_idx = progress.get("current", 0)
        is_resuming = current_idx > 0 and current_idx < word_count
        current_word = session_words[current_idx] if session_words and current_idx < word_count else None
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
                        stop_secs=5.0,  # 5s silence → child is done spelling
                    )
                ),
                audio_out_10ms_chunks=20,
            ),
        )

        llm_base = os.getenv("NVIDIA_LLM_URL", VLLM_VL_BASE)
        llm_model = os.getenv("NVIDIA_LLM_MODEL", VLLM_VL_MODEL)
        llm = NvidiaLLMService(
            api_key="not-needed",
            base_url=llm_base,
            model=llm_model,
            params=NvidiaLLMService.InputParams(
                max_tokens=80,
                temperature=0.1,
                extra={"stop": ["Nouns:", "Verbs:", "Adjectives:", "Anagrams:"]},
            ),
        )

        stt = ElevenLabsRealtimeSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            sample_rate=16000,
            params=ElevenLabsRealtimeSTTService.InputParams(
                language_code="eng",
                # Children pause 2-3s between letters when spelling aloud.
                # Default 1.5s finalizes the transcript too early, causing the
                # LLM to respond before the child finishes spelling.
                vad_silence_threshold_secs=4.0,
            ),
        )

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_TTS_VOICE_ID", "3vbrfmIQGJrswxh7ife4"),
            model="eleven_turbo_v2_5",
            sample_rate=16000,
        )

        stt_transcript_synchronization = UserTranscriptSynchronization()
        tts_transcript_synchronization = BotTranscriptSynchronization()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a spelling bee quiz host for children. "
                    "No markdown. Plain text only.\n\n"

                    "ABSOLUTE RULE: You are ONLY a spelling bee coach. You know NOTHING "
                    "about weather, math, science, news, or ANY topic outside spelling. "
                    "If the child says ANYTHING that is not about spelling (asking for "
                    "weather, telling a story, asking questions, etc.), you MUST respond "
                    "ONLY with: 'Let us get back to spelling practice. Your word is [current word].' "
                    "NEVER answer off-topic questions. NEVER say 'I don't have access to' anything. "
                    "NEVER apologize for not being able to help with something. Just redirect.\n\n"

                    "YOUR JOB: Present ONE word, wait for the child to spell it, judge, "
                    "then present the next word. NEVER handle more than one word per turn. "
                    "NEVER generate content for multiple words at once.\n\n"

                    "GREETING:\n"
                    "- When the user says 'Start the spelling bee', say EXACTLY: "
                    "'Hello! Welcome to spelling bee practice. Your first word is [first word].'\n"
                    "- When the user says 'Continue the spelling bee', say EXACTLY: "
                    "'Welcome back! Let us continue our spelling bee practice. Your word is [current word].'\n\n"

                    "PRESENTING WORDS:\n"
                    "- Say ONLY the whole word. Example: 'Your word is elephant.'\n"
                    "- NEVER spell out, break apart, or show letters. FORBIDDEN output: "
                    "C A L E N D A R, B-E-A-U-T-I-F-U-L, 'e, l, e, p, h, a, n, t'.\n"
                    "- NEVER repeat the correct spelling letter by letter for any reason.\n"
                    "- If the child asks for a sentence, give ONE short sentence using the word, "
                    "then repeat the word. NEVER give sentences for multiple words.\n"
                    "- If the child asks for a definition or meaning, give ONE short "
                    "child-friendly definition. Never list multiple meanings.\n\n"

                    "SPELLING VERIFICATION:\n"
                    "You will receive internal system messages with a verdict. "
                    "Use the verdict to respond. NEVER reveal, recap, quote, or "
                    "paraphrase the system message. NEVER show the child's spelling "
                    "letter by letter. NEVER show a 'recap' or 'summary'.\n\n"

                    "RESPONSE FORMAT — MANDATORY (you MUST follow this EXACTLY):\n"
                    "- Say ONLY one of these two patterns and NOTHING ELSE:\n"
                    "  Pattern A: 'Correct! Your next word is [word].'\n"
                    "  Pattern B: 'Not quite. The correct spelling is [word]. Your next word is [word].'\n"
                    "- NEVER add ANY additional text, explanation, or commentary.\n"
                    "- NEVER say 'That is correct', 'Great', 'Good job', 'Well done', "
                    "'The spelling is...', 'You spelled...', 'The child spelled...', "
                    "'Here is a recap', or ANY variation. ONLY 'Correct!' or 'Not quite.'\n"
                    "- NEVER use markdown, newlines, bullet points, or emojis.\n"
                    "- Your ENTIRE response must be ONE short sentence. STOP immediately "
                    "after announcing the next word. Do NOT generate a second sentence.\n"
                    "- NEVER provide example sentences, definitions, or extra commentary "
                    "unless the child explicitly asks.\n"
                    "- After announcing a word, STOP. Do NOT continue generating text. "
                    "WAIT for the child to spell it before responding.\n"
                    "- IMPORTANT: Whether correct or wrong, ALWAYS move to the NEXT word "
                    "in the list. NEVER repeat the same word. Missed words will be "
                    "reviewed at the end.\n"
                    "- On repeat request: 'Your word is [same word].'\n"
                    "- On skip request: 'Your next word is [next word].'\n"
                    "- After the last word, if correct: 'Correct! All done!'\n"
                    "- After the last word, if wrong: 'Not quite. The correct spelling is [word]. All done!'\n"
                    "- After saying 'All done!' ALWAYS check: if the child got ANY "
                    "words wrong during this session, AUTOMATICALLY start reviewing "
                    "by saying: 'Now let's review the words you missed. Your word is "
                    "[first missed word].' then STOP and WAIT for the child to spell it. "
                    "Do NOT ask if they want to review. Do NOT judge until the child responds.\n"
                    "- If the child says 'I want to review' or 'review' at any time, "
                    "re-quiz ONLY the words they got wrong so far, in the same format.\n"
                    "- During the review round, re-quiz ONLY the missed words "
                    "in the same format. After the review round completes, say: "
                    "'Great practice today! Keep it up!' Do NOT start another review round.\n"
                    "- If the child got everything correct, skip review and say: "
                    "'Great practice today! Keep it up!'\n\n"

                    "STAY ON TOPIC (CRITICAL — HIGHEST PRIORITY):\n"
                    "- ONLY discuss spelling. You have NO knowledge of any other topic.\n"
                    "- If the child asks about weather, math, stories, games, or ANYTHING "
                    "other than spelling: say ONLY 'Let us get back to spelling practice. "
                    "Your word is [current word].' and STOP.\n"
                    "- NEVER answer, acknowledge, or apologize for off-topic requests.\n"
                    "- NEVER say 'I don't have access to', 'I can't help with', or similar.\n\n"

                    f"Total words: {word_count}.\n"
                    + (
                        "WORD LIST (quiz in this exact order, NEVER read the list aloud):\n"
                        + ", ".join(session_words)
                        + "\n"
                        if session_words else "No words uploaded yet.\n"
                    )
                ),
            }
        ]

        context = LLMContext(messages)
        context_aggregator = create_nvidia_context_aggregator(context, send_interims=False)

        md_stripper = MarkdownStripper()
        transcript_bridge = TranscriptBridge()
        user_transcript_bridge = UserTranscriptBridge()
        guardrails_filter = GuardrailsFilter(messages)
        review_injector = ReviewInjector(session_id, word_count, messages)
        spelling_verifier = SpellingVerifier(session_id, session_words, review_injector, messages)
        incorrect_tracker = IncorrectWordTracker(session_id, session_words, review_injector, spelling_verifier)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                user_transcript_bridge,
                stt_transcript_synchronization,
                guardrails_filter,
                review_injector,
                spelling_verifier,
                context_aggregator.user(),
                llm,
                incorrect_tracker,
                md_stripper,
                tts,
                tts_transcript_synchronization,
                transcript_bridge,
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
            if session_words and is_resuming:
                messages.append(
                    {
                        "role": "user",
                        "content": "Continue the spelling bee.",
                    }
                )
            elif session_words:
                messages.append(
                    {
                        "role": "user",
                        "content": "Start the spelling bee.",
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
