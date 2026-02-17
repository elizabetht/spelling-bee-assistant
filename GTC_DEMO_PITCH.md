# Spelling Bee Assistant — NVIDIA GTC Golden Ticket Demo Pitch

## Title

**Spelling Bee Assistant: Real-Time AI Voice Coaching for Kids, Powered by NVIDIA**

---

## The Problem

Every week, millions of elementary school students bring home a spelling word list stapled to a worksheet. Practice usually means a parent reading words aloud while the child spells them back — repetitive for the parent, stressful for the child, and impossible when the parent isn't available. Existing spelling apps are text-based, tap-to-type experiences that don't simulate the real spelling bee format: *hearing* a word, *saying* the letters aloud, and getting immediate spoken feedback.

**What if a child could just snap a photo of their word list and instantly have a patient, encouraging AI coach run an interactive voice spelling bee — anytime, anywhere?**

---

## The Solution

**Spelling Bee Assistant** is a fully voice-driven AI coaching app that turns any word list image into a live, interactive spelling practice session. The child uploads a photo of their weekly word list, and within seconds a friendly AI voice coach begins the session — pronouncing each word, offering definitions and example sentences on request, listening to the child spell aloud letter by letter, and providing real-time feedback with encouragement.

The app tracks correct and incorrect answers, and at the end of the session offers a **review round** for missed words — just like a real spelling bee coach would.

---

## How It Works (Demo Flow)

### Step 1 — Upload the Word List
The child (or parent) takes a photo of the printed spelling word list and uploads it through the browser. The **NVIDIA Nemotron-Nano-12B-VL-FP8 vision-language model**, self-hosted via **vLLM** on an **NVIDIA DGX Spark (GB10 GPU)**, reads the image and extracts the words — no manual typing needed.

### Step 2 — Start the Voice Session
One click starts a real-time voice session over WebSocket. The **NVIDIA Pipecat (ACE) pipeline** orchestrates the entire audio loop:

1. The AI coach announces: *"Your first word is 'journey.' Would you like a sentence or definition?"*
2. The child responds naturally — *"Can I hear it in a sentence?"*
3. **NVIDIA Riva ASR** (streaming speech-to-text) transcribes the child's voice in real time
4. The transcription flows through a **server-side spelling verifier** that parses letter sequences (e.g. "J-O-U-R-N-E-Y") and determines correctness deterministically
5. **NeMo Guardrails** enforce child-safe, on-topic conversation — the LLM won't discuss anything outside spelling practice
6. **Nemotron-Nano** generates a warm, contextual response with the verdict
7. **ElevenLabs TTS** speaks the response back in a natural, friendly voice

### Step 3 — Score Tracking & Review
The browser UI shows a live score bar: round number, correct/incorrect counts, and progress through the list. At the end:
- If the child has missed any words, the coach offers a **review round** for just those words
- Incorrect words are stored in **Redis** and re-quizzed until mastered
- The session ends with encouragement and a summary

---

## NVIDIA Technologies Used

| Technology | Role in the App |
|---|---|
| **NVIDIA Pipecat (ACE)** | Real-time voice agent pipeline framework — orchestrates the full audio-in → ASR → LLM → TTS → audio-out loop with sub-second latency |
| **NVIDIA Riva ASR** | Streaming speech-to-text with low latency — optimized for real-time voice transcription |
| **Nemotron-Nano-12B-VL-FP8** | Dual-purpose vision-language model: (1) extracts spelling words from uploaded images, (2) powers the conversational spelling coach — both served from a single self-hosted vLLM instance |
| **NeMo Guardrails** | Programmable safety rails that enforce child-appropriate, spelling-only conversation — blocks off-topic requests and ensures age-appropriate content |
| **NVIDIA Container Runtime** | GPU-accelerated container execution on Kubernetes for the vLLM model serving pod |
| **NVIDIA DGX Spark (GB10)** | Edge GPU hardware running the self-hosted vLLM model — demonstrates that a complete AI voice agent can run on affordable edge infrastructure |

---

## Additional Technologies

| Technology | Role |
|---|---|
| **vLLM** | High-throughput model serving with FP8 quantization on GB10 |
| **ElevenLabs TTS** | Natural, friendly voice synthesis for the AI coach |
| **FastAPI + WebSocket** | Low-latency backend for REST image upload and real-time audio transport |
| **Redis** | Session state: word lists, progress, incorrect word tracking (24h TTL) |
| **Tesseract OCR** | Fallback word extraction when the VLM is unavailable |
| **Kubernetes (microk8s)** | Multi-node GPU-aware deployment with controller + GPU node topology |

---

## What Makes This Special

- **Voice-first, not text-first** — simulates a real spelling bee, not a typing quiz
- **Zero manual input** — snap a photo of any word list and go
- **One model, two jobs** — Nemotron-Nano handles both image understanding and conversational coaching
- **Server-side spelling verification** — deterministic letter-by-letter comparison ensures accurate scoring even when the LLM is uncertain
- **Child-safe by design** — NeMo Guardrails keep every conversation on-topic and age-appropriate
- **Edge-deployable** — runs entirely on a single NVIDIA DGX Spark, proving that sophisticated voice AI doesn't require a data center
- **Review loop** — missed words are automatically re-quizzed, reinforcing learning

---

## Demo Script (2–3 minutes)

1. **[Show the UI]** "This is Spelling Bee Assistant. A child uploads a photo of their weekly word list..."
2. **[Upload image]** "Nemotron-Nano's vision-language model reads the image and extracts 25 words in under 3 seconds."
3. **[Start session]** "Now watch — the AI coach begins the spelling bee entirely by voice."
4. **[Spell a word correctly]** "The child says J-O-U-R-N-E-Y, and the coach responds: 'Correct! Great job!'"
5. **[Spell a word incorrectly]** "Now they say E-X-E-R-C-S-E... the server-side verifier catches the missing 'I', and the coach says: 'Not quite. The correct spelling is exercise.'"
6. **[Ask for a sentence]** "The child asks 'Can I hear it in a sentence?' — Nemotron generates one on the fly."
7. **[Show score bar]** "The UI tracks every response — Round 3, 2 correct, 1 incorrect."
8. **[End & review]** "After all words, the coach offers to review the missed ones. One more chance to get them right."
9. **[Wrap up]** "The entire pipeline — vision, voice, LLM, guardrails — runs on a single NVIDIA DGX Spark. That's the power of NVIDIA's full-stack AI platform."

---

## One-Liner

> *Snap a photo of your spelling list, and an AI voice coach powered by NVIDIA Nemotron + Pipecat ACE runs an interactive spelling bee — entirely by voice, entirely on the edge.*
