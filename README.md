# Spelling Bee Assistant

**An AI-powered voice coaching application that turns any word list image into an interactive spelling practice session using NVIDIA's full-stack AI platform.**

Students upload a photo of their spelling word list, and the assistant extracts the words using a vision-language model, then conducts a real-time voice-driven spelling quiz with pronunciation, definitions, example sentences, and encouragement — all guarded by NeMo Guardrails to keep the conversation child-safe and on-topic.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER (Browser)                                 │
│                                                                             │
│   ┌─────────────────────┐              ┌──────────────────────────────┐     │
│   │   Upload Word List  │              │   Voice Spelling Session     │     │
│   │   Image (REST)      │              │   (WebSocket Audio)          │     │
│   └────────┬────────────┘              └──────────────┬───────────────┘     │
└────────────┼──────────────────────────────────────────┼─────────────────────┘
             │                                          │
             ▼                                          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend (:8080)                              │
│                                                                            │
│   ┌────────────────────┐         ┌───────────────────────────────────┐     │
│   │  POST /upload-image│         │  WS /pipecat/ws                  │     │
│   │                    │         │                                   │     │
│   │  1. Decode image   │         │  Pipecat ACE Pipeline            │     │
│   │  2. Extract words  │         │  ┌─────────────────────────────┐ │     │
│   │     via VLM        │         │  │                             │ │     │
│   │  3. Store in Redis │         │  │  Audio In ──► Riva ASR      │ │     │
│   │  4. Return         │         │  │                 │            │ │     │
│   │     session_id     │         │  │                 ▼            │ │     │
│   │                    │         │  │         NeMo Guardrails      │ │     │
│   └────────┬───────────┘         │  │                 │            │ │     │
│            │                     │  │                 ▼            │ │     │
│            │                     │  │          NVIDIA LLM          │ │     │
│            │                     │  │       (Spelling Coach)       │ │     │
│            │                     │  │                 │            │ │     │
│            │                     │  │                 ▼            │ │     │
│            │                     │  │           Riva TTS           │ │     │
│            │                     │  │                 │            │ │     │
│            │                     │  │                 ▼            │ │     │
│            │                     │  │           Audio Out          │ │     │
│            │                     │  │                             │ │     │
│            │                     │  └─────────────────────────────┘ │     │
│            │                     └───────────────────────────────────┘     │
│            │                                    │                          │
│            ▼                                    ▼                          │
│   ┌─────────────────────────────────────────────────────┐                  │
│   │                   Redis                             │                  │
│   │   Session words, progress, chat history (24h TTL)   │                  │
│   └─────────────────────────────────────────────────────┘                  │
└────────────────────────────────────────────────────────────────────────────┘
             │                                    │
             ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          NVIDIA AI Services                                │
│                                                                            │
│   ┌──────────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│   │  Nemotron-Nano   │  │  Riva ASR    │  │  Riva TTS                  │  │
│   │  VL-8B (vLLM)    │  │  Parakeet    │  │  Magpie-Multilingual Sofia │  │
│   │                  │  │  1.1B        │  │                            │  │
│   │  Image → Words   │  │  Speech →    │  │  Text → Speech             │  │
│   │  Definitions     │  │  Text        │  │                            │  │
│   │  Sentences       │  │              │  │                            │  │
│   └──────────────────┘  └──────────────┘  └────────────────────────────┘  │
│                                                                            │
│   ┌──────────────────┐  ┌──────────────────────────────────────────────┐  │
│   │  NVIDIA LLM      │  │  NeMo Guardrails                            │  │
│   │  Llama-3.1-8B    │  │  Topic enforcement, intent filtering,       │  │
│   │  Instruct        │  │  child-safe content policy                   │  │
│   └──────────────────┘  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

## Component Summary

| Component | Technology | Role |
|---|---|---|
| **Voice Pipeline** | NVIDIA Pipecat (ACE) | Orchestrates real-time audio I/O, ASR, LLM, and TTS |
| **Vision-Language Model** | Nemotron-Nano-VL-8B via vLLM | Extracts spelling words from uploaded images |
| **Speech Recognition** | Riva ASR (Parakeet 1.1B) | Streaming speech-to-text with VAD |
| **Text-to-Speech** | Riva TTS (Magpie Sofia) | Natural voice output for the spelling coach |
| **Conversational LLM** | Llama-3.1-8B-Instruct | Powers the interactive spelling coach persona |
| **Safety** | NeMo Guardrails | Enforces spelling-only scope, filters off-topic intent |
| **Session Store** | Redis + LangChain | Persistent word lists, progress, and chat history |
| **Fallback OCR** | Tesseract (pytesseract) | Backup word extraction when VLM is unavailable |
| **Web Framework** | FastAPI + Uvicorn | REST API, WebSocket transport, static UI |
| **Orchestration** | Kubernetes (microk8s) | Multi-node GPU-aware deployment |

## User Flow

```
 ┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  Upload  │     │  Words       │     │  Voice       │     │  Interactive │
 │  word    │────►│  extracted   │────►│  session     │────►│  spelling    │
 │  list    │     │  via VLM     │     │  begins      │     │  practice    │
 │  image   │     │  + stored    │     │  (WebSocket) │     │  with coach  │
 └──────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                    │
                                              ┌─────────────────────┤
                                              ▼                     ▼
                                       "Use it in a          "What does it
                                        sentence?"             mean?"
                                              │                     │
                                              ▼                     ▼
                                        VLM generates         VLM generates
                                        child-friendly        age-appropriate
                                        sentence              definition
```

**During a session, the student can:**
- Hear the word pronounced
- Ask for it in a sentence
- Request a definition
- Spell the word aloud and receive feedback
- Skip to the next word
- Receive encouragement throughout

## Deployment Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster (microk8s)                  │
│                        Namespace: spellingbee                         │
│                                                                      │
│   ┌─────────────────────────────────────┐                            │
│   │         Controller Node             │                            │
│   │                                     │                            │
│   │   ┌───────────────────────────┐     │                            │
│   │   │  Backend Pod              │     │                            │
│   │   │  spelling-bee-agent       │     │   ┌──────────────────────┐ │
│   │   │  NodePort :30088          │     │   │    GPU Node          │ │
│   │   └───────────┬───────────────┘     │   │                      │ │
│   │               │                     │   │   ┌────────────────┐ │ │
│   │   ┌───────────▼───────────────┐     │   │   │  vLLM Pod      │ │ │
│   │   │  Redis Pod                │     │   │   │  Nemotron-Nano │ │ │
│   │   │  Session Store            │     │   │   │  VL-8B         │ │ │
│   │   └───────────────────────────┘     │   │   │  NodePort      │ │ │
│   │                                     │   │   │  :30566        │ │ │
│   └─────────────────────────────────────┘   │   │  GPU: GB10     │ │ │
│                                              │   └────────────────┘ │ │
│                                              └──────────────────────┘ │
│                                                                      │
│   External (NVIDIA Cloud):                                           │
│     • Riva ASR  → grpc.nvcf.nvidia.com:443                          │
│     • Riva TTS  → grpc.nvcf.nvidia.com:443                          │
│     • LLM API   → integrate.api.nvidia.com/v1                       │
└──────────────────────────────────────────────────────────────────────┘
```

## NVIDIA Technologies Used

- **NVIDIA Pipecat (ACE)** — Real-time voice agent pipeline framework
- **Riva ASR** — Automatic speech recognition (Parakeet 1.1B, streaming with Silero VAD)
- **Riva TTS** — Text-to-speech synthesis (Magpie-Multilingual Sofia EN-US)
- **Nemotron-Nano-VL-8B** — Vision-language model for image understanding, served via vLLM
- **NeMo Guardrails** — Programmable safety rails for topic enforcement and content filtering
- **NVIDIA LLM API** — Llama-3.1-8B-Instruct for conversational coaching
- **NVIDIA Container Runtime** — GPU-accelerated container execution

## Getting Started

**Requirements:** Python 3.12+ (required by NVIDIA Pipecat)

```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables
export NVIDIA_API_KEY=<your-key>
export RIVA_ASR_URL=grpc.nvcf.nvidia.com:443
export RIVA_TTS_URL=grpc.nvcf.nvidia.com:443

# Optional: enable guardrails
export ENABLE_NEMO_GUARDRAILS=true
export NEMO_GUARDRAILS_CONFIG_PATH=./guardrails

# Start the server
python spelling_bee_agent_backend.py
```

Open `http://localhost:8080` to access the test UI.

## Deploy to Kubernetes

```bash
# Deploy vLLM model + backend together
./scripts/deploy_all.sh

# Or separately
./scripts/deploy_model.sh      # vLLM vision-language model (GPU node)
./scripts/deploy_backend.sh    # FastAPI backend (controller node)

# With secret creation
CREATE_SECRET=true NVIDIA_API_KEY_VALUE=<key> ./scripts/deploy_backend.sh
```

## Smoke Test

```bash
./scripts/smoke_test.sh http://localhost:8080 ./path/to/words-image.png
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Test UI |
| `/healthz` | GET | Health check (reports Pipecat availability) |
| `/upload-image` | POST | Upload word list image, returns `session_id` |
| `/pipecat/ws` | WebSocket | Voice session — connect with `?session_id=<id>` |

## Project Structure

```
spelling-bee-assistant/
├── spelling_bee_agent_backend.py   # FastAPI backend + Pipecat pipeline
├── ui/
│   └── index.html                  # Browser-based test UI
├── guardrails/
│   ├── config.yml                  # NeMo Guardrails model config
│   └── rails.co                    # Intent policies (spelling scope)
├── scripts/
│   ├── deploy_all.sh               # Deploy model + backend
│   ├── deploy_backend.sh           # Deploy backend only
│   ├── deploy_model.sh             # Deploy vLLM model only
│   └── smoke_test.sh               # End-to-end integration test
├── spelling-bee-agent-backend.k8s.yaml  # K8s backend manifest
├── vllm-nemotron-nano-vl-8b.yaml        # K8s vLLM model manifest
├── Dockerfile                      # Backend container image
└── requirements.txt                # Python dependencies
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | — | NVIDIA API authentication (required) |
| `RIVA_ASR_URL` | — | Riva ASR endpoint (required) |
| `RIVA_TTS_URL` | — | Riva TTS endpoint (required) |
| `ENABLE_NEMO_GUARDRAILS` | `false` | Enable NeMo Guardrails |
| `NEMO_GUARDRAILS_CONFIG_PATH` | `./guardrails` | Path to guardrails config |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `VLLM_VL_BASE` | `http://vllm-nemotron-nano-vl-8b:5566/v1` | vLLM endpoint |
| `VLLM_VL_MODEL` | `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` | Vision-language model |
| `NVIDIA_LLM_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA LLM API |
| `NVIDIA_LLM_MODEL` | `meta/llama-3.1-8b-instruct` | Conversational LLM |
