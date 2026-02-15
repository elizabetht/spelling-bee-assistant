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
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import redis
from langchain.memory import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

# Placeholder for NeMo Guardrails integration
# from nemoguardrails import Rails

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for web UI)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Redis setup for Langchain memory
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL)

# Session state (for demo, use Redis in production)
session_wordlists = {}  # session_id: [words]
session_progress = {}   # session_id: {current, incorrect, skipped}

# --- OCR/vision-language model for extracting words from image ---
def extract_words_from_image(image_bytes) -> List[str]:
    image = Image.open(image_bytes)
    text = pytesseract.image_to_string(image)
    # Simple split: one word per line, filter out non-alpha
    words = [w.strip() for w in text.splitlines() if w.strip().isalpha()]
    return words

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    session_id = os.urandom(8).hex()
    words = extract_words_from_image(file.file)
    session_wordlists[session_id] = words
    session_progress[session_id] = {"current": 0, "incorrect": [], "skipped": []}
    return {"session_id": session_id, "words": words}

# --- WebSocket for quiz interaction ---
@app.websocket("/ws/quiz/{session_id}")
async def quiz_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    words = session_wordlists.get(session_id, [])
    progress = session_progress.get(session_id, {"current": 0, "incorrect": [], "skipped": []})
    # Langchain memory setup
    chat_history = RedisChatMessageHistory(session_id, url=REDIS_URL)
    memory = ConversationBufferMemory(chat_memory=chat_history)
    try:
        await websocket.send_text("Welcome to the Spelling Bee! Say 'start' to begin.")
        while True:
            data = await websocket.receive_text()
            # --- NeMo Guardrails check would go here ---
            # if not rails.is_allowed(data):
            #     await websocket.send_text("Let's focus on spelling bee practice!")
            #     continue
            if data.lower() == "start":
                idx = progress["current"]
                if idx < len(words):
                    await websocket.send_text(f"Spell the word: {words[idx]}")
                else:
                    await websocket.send_text("No more words. Well done!")
                    break
            elif data.lower() == "repeat":
                idx = progress["current"]
                await websocket.send_text(f"Repeat: {words[idx]}")
            elif data.lower() == "skip":
                idx = progress["current"]
                progress["skipped"].append(words[idx])
                progress["current"] += 1
                await websocket.send_text("Word skipped. Say 'start' for next word.")
            elif data.lower() == "sentence":
                idx = progress["current"]
                # Placeholder: call VL model for sentence
                sentence = f"Example: I can spell the word {words[idx]} easily."
                await websocket.send_text(sentence)
            elif data.lower() == "definition":
                idx = progress["current"]
                # Placeholder: call VL model for definition
                definition = f"Definition of {words[idx]}: (definition here)"
                await websocket.send_text(definition)
            elif data.lower().startswith("spell "):
                guess = data[6:].strip().lower()
                idx = progress["current"]
                correct = words[idx].lower()
                if guess == correct:
                    await websocket.send_text("Correct! Great job! Say 'start' for next word.")
                    progress["current"] += 1
                else:
                    await websocket.send_text("That's not quite right. Try again or say 'skip'.")
                    progress["incorrect"].append(guess)
            elif data.lower() == "review":
                incorrect = progress["incorrect"]
                await websocket.send_text(f"Incorrect words so far: {incorrect}")
            elif data.lower() == "resume":
                idx = progress["current"]
                await websocket.send_text(f"Resuming at word: {words[idx]}")
            else:
                await websocket.send_text("Allowed commands: start, repeat, skip, sentence, definition, review, resume, spell <word>")
            # Save progress
            session_progress[session_id] = progress
    except WebSocketDisconnect:
        pass

# --- Static HTML for quick test (replace with real UI) ---
@app.get("/")
async def root():
    return HTMLResponse("""
    <html><body>
    <h2>Spelling Bee Assistant</h2>
    <form action='/upload-image' enctype='multipart/form-data' method='post'>
      <input name='file' type='file'/><input type='submit'/>
    </form>
    <p>After uploading, connect to <code>/ws/quiz/&lt;session_id&gt;</code> via WebSocket.</p>
    </body></html>
    """)
