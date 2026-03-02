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
│   │  3. Store in Redis │         │  │  Audio In ──► ElevenLabs   │ │     │
│   │  4. Return         │         │  │              ASR (Scribe)   │ │     │
│   │     session_id     │         │  │                 ▼            │ │     │
│   │                    │         │  │         NeMo Guardrails      │ │     │
│   └────────┬───────────┘         │  │                 │            │ │     │
│            │                     │  │                 ▼            │ │     │
│            │                     │  │        Nemotron-Nano         │ │     │
│            │                     │  │       (Spelling Coach)       │ │     │
│            │                     │  │                 │            │ │     │
│            │                     │  │                 ▼            │ │     │
│            │                     │  │         ElevenLabs TTS        │ │     │
│            │                     │  │         (Cloud API)           │ │     │
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
│   ┌───────────────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │  Nemotron-Nano-12B-VL-FP8│  │ ElevenLabs   │  │ ElevenLabs TTS   │  │
│   │  (vLLM, self-hosted)     │  │ STT (Scribe) │  │ (Cloud API)      │  │
│   │                          │  │ (cloud API)  │  │                  │  │
│   │  Image → Words           │  │ Speech →     │  │  Text → Speech   │  │
│   │  Definitions / Sentences │  │ Text         │  │  16kHz PCM       │  │
│   │  Voice Coach LLM         │  │              │  │                  │  │
│   └───────────────────────────┘  └──────────────┘  └──────────────────┘  │
│                                                                            │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  NeMo Guardrails                                                │  │
│   │  Topic enforcement, intent filtering, child-safe content policy │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

## Component Summary

| Component | Technology | Role |
|---|---|---|
| **Voice Pipeline** | NVIDIA Pipecat (ACE) | Orchestrates real-time audio I/O, ASR, LLM, and TTS |
| **Vision-Language Model** | Nemotron-Nano-12B-VL-FP8 via vLLM | Extracts spelling words from uploaded images |
| **Speech Recognition** | ElevenLabs STT (Scribe, cloud API) | Streaming speech-to-text via WebSocket |
| **Text-to-Speech** | ElevenLabs TTS (Cloud API) | Natural voice output at 16kHz |
| **Conversational LLM** | Nemotron-Nano-12B-VL-FP8 via vLLM | Powers the interactive spelling coach (same model as VLM) |
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
                                        LLM generates         LLM generates
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
│   │   │  Session Store            │     │   │   │  12B-VL-FP8    │ │ │
│   │   └───────────────────────────┘     │   │   │  NodePort      │ │ │
│   │                                     │   │   │  :30566        │ │ │
│   └─────────────────────────────────────┘   │   │  GPU: GB10     │ │ │
│                                              │   └────────────────┘ │ │
│                                              └──────────────────────┘ │
│                                                                      │
│   External (Cloud APIs):                                             │
│     • ElevenLabs ASR  → api.elevenlabs.io (Scribe v1)                │
│     • ElevenLabs TTS  → api.elevenlabs.io (Cloud API)                │
└──────────────────────────────────────────────────────────────────────┘
```

## NVIDIA Technologies Used

- **NVIDIA Pipecat (ACE)** — Real-time voice agent pipeline framework
- **ElevenLabs STT (Scribe)** — Cloud-hosted streaming speech-to-text (WebSocket API)
- **ElevenLabs TTS** — Cloud-hosted text-to-speech (Cloud API)
- **Nemotron-Nano-12B-VL-FP8** — Vision-language model for image understanding and conversational coaching, served via vLLM
- **NeMo Guardrails** — Programmable safety rails for topic enforcement and content filtering
- **NVIDIA Container Runtime** — GPU-accelerated container execution

## Getting Started

**Requirements:** Python 3.12+ (required by NVIDIA Pipecat)

```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables
export ELEVENLABS_API_KEY=<your-key>      # For ElevenLabs ASR + TTS

# Optional: enable guardrails
export ENABLE_NEMO_GUARDRAILS=true
export NEMO_GUARDRAILS_CONFIG_PATH=./guardrails

# Start the server
python spelling_bee_agent_backend.py
```

Open `http://localhost:8080` to access the test UI.

## Launch with Brev

The easiest way to deploy this application is using **Brev**, NVIDIA's platform for launching AI applications.

**Prerequisites:**
- Brev account and CLI installed
- ElevenLabs API key

**Launch Steps:**

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/elizabetht/spelling-bee-assistant.git
   cd spelling-bee-assistant
   ```

2. **Set your ElevenLabs API key**:
   ```bash
   brev secret set ELEVENLABS_API_KEY=your-key-here
   ```

3. **Launch the application**:
   ```bash
   brev launch
   ```
   
   Or via the Brev dashboard:
   - Navigate to your Brev dashboard
   - Click "New Launch"
   - Select this repository
   - Configure the `ELEVENLABS_API_KEY` secret
   - Click "Launch"

4. **Access your application**:
   - Brev will provide a URL (typically `https://your-app.brev.dev`)
   - Open the URL in your browser to access the spelling bee assistant
   - Upload a word list image to start a session

**Configuration Options:**

All optional environment variables can be set via Brev secrets or the `.brev.yaml` configuration:
- `VLLM_VL_BASE` - Custom vLLM endpoint (default: auto-configured)
- `ENABLE_NEMO_GUARDRAILS` - Enable content safety (default: `false`)
- `REDIS_URL` - Custom Redis URL (default: uses Brev-managed Redis)

For more details, see [.brev/README.md](.brev/README.md).

## Deploy to Kubernetes

Pre-requisites: a microk8s cluster with the `spellingbee` namespace, a local
container registry at `localhost:32000`, and GPU nodes with the NVIDIA runtime.

**1. Create secrets**

```bash
# ElevenLabs API key (required for ASR + TTS)
kubectl -n spellingbee create secret generic elevenlabs-api-key \
  --from-literal=api-key=<YOUR_ELEVENLABS_KEY>

# HuggingFace token (required for vLLM model download)
kubectl -n spellingbee create secret generic hf-token \
  --from-literal=token=<YOUR_HF_TOKEN>
```

**2. Deploy everything (model + Redis + backend)**

```bash
./deploy/deploy_all.sh
```

Or deploy individually:

```bash
./deploy/deploy_model.sh      # vLLM Nemotron-Nano-12B-VL-FP8 on GPU node
./deploy/deploy_redis.sh      # Redis session store on controller node
./deploy/deploy_backend.sh    # FastAPI backend on controller node
```

> **Note:** ASR and TTS are cloud-hosted via ElevenLabs, so no GPU pod for
> speech services is needed. Only the vLLM model requires a GPU.

The backend script builds the Docker image, pushes it to the local registry,
applies the K8s manifest, and waits for rollout.

**3. Verify**

```bash
kubectl -n spellingbee get pods -o wide
kubectl -n spellingbee get svc
```

**4. Smoke test**

```bash
./deploy/smoke_test.sh http://<controller-ip>:30088 ./path/to/words-image.png
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
├── deploy/
│   ├── spelling-bee-agent-backend.k8s.yaml  # K8s backend manifest
│   ├── vllm-nemotron-nano-vl-8b.yaml        # K8s vLLM model manifest
│   ├── redis.k8s.yaml                       # K8s Redis manifest
│   ├── deploy_all.sh               # Deploy model + Redis + backend
│   ├── deploy_backend.sh           # Deploy backend only
│   ├── deploy_model.sh             # Deploy vLLM model only
│   ├── deploy_redis.sh             # Deploy Redis only
│   └── smoke_test.sh               # End-to-end integration test
├── Dockerfile                      # Backend container image
└── requirements.txt                # Python dependencies
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ELEVENLABS_API_KEY` | — | ElevenLabs API key (required for ASR + TTS) |
| `ELEVENLABS_TTS_VOICE_ID` | `3vbrfmIQGJrswxh7ife4` | ElevenLabs TTS voice identifier |
| `ENABLE_NEMO_GUARDRAILS` | `false` | Enable NeMo Guardrails |
| `NEMO_GUARDRAILS_CONFIG_PATH` | `./guardrails` | Path to guardrails config |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `VLLM_VL_BASE` | `http://vllm-nemotron-nano-vl-8b:5566/v1` | vLLM endpoint (used for both image extraction and voice coaching) |
| `VLLM_VL_MODEL` | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8` | Vision-language model (one model, two roles) |
| `NVIDIA_LLM_URL` | Same as `VLLM_VL_BASE` | Override LLM endpoint for voice pipeline |
| `NVIDIA_LLM_MODEL` | Same as `VLLM_VL_MODEL` | Override LLM model for voice pipeline |
