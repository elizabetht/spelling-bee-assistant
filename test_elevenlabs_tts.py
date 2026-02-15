#!/usr/bin/env python3
"""Quick test: generate speech with ElevenLabs TTS and play it."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_TTS_VOICE_ID", "9BWtsMINqrJLrRacOk9x")  # Aria
MODEL = "eleven_turbo_v2_5"
TEXT = "Hello! Welcome to Spelling Bee practice. Your first word is: adventure."

if not API_KEY:
    sys.exit("ELEVENLABS_API_KEY not set. Export it or add to .env")

print(f"Voice ID : {VOICE_ID}")
print(f"Model    : {MODEL}")
print(f"Text     : {TEXT}")
print()

# --- Method 1: ElevenLabs SDK (pip install elevenlabs) ---
try:
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=API_KEY)

    print("Generating audio via ElevenLabs SDK...")
    audio_iter = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        model_id=MODEL,
        text=TEXT,
    )

    # Collect all chunks
    audio_bytes = b"".join(audio_iter)
    out_file = "test_output.mp3"
    with open(out_file, "wb") as f:
        f.write(audio_bytes)

    print(f"Saved {len(audio_bytes)} bytes -> {out_file}")
    print(f"Play it:  open {out_file}   (macOS)  or  aplay {out_file}  (Linux)")

except ImportError:
    print("elevenlabs SDK not installed. pip install elevenlabs")
except Exception as e:
    print(f"SDK error: {e}")

# --- Method 2: Raw REST API (no extra deps) ---
print("\n--- Trying raw REST API ---")
try:
    import urllib.request
    import json

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    payload = json.dumps({
        "text": TEXT,
        "model_id": MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )

    print("Calling ElevenLabs REST API...")
    with urllib.request.urlopen(req, timeout=15) as resp:
        audio_bytes = resp.read()

    out_file2 = "test_output_rest.mp3"
    with open(out_file2, "wb") as f:
        f.write(audio_bytes)
    print(f"Saved {len(audio_bytes)} bytes -> {out_file2}")
    print(f"Play it:  open {out_file2}")

except Exception as e:
    print(f"REST API error: {e}")
