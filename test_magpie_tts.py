#!/usr/bin/env python3
"""Quick test: generate speech with NVIDIA MagpieTTS and save it."""

import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
VOICE_ID = os.getenv("NVIDIA_TTS_VOICE_ID", "Magpie-Multilingual.EN-US.Aria")
SERVER = os.getenv("NVIDIA_TTS_SERVER", "grpc.nvcf.nvidia.com:443")
FUNCTION_ID = os.getenv("NVIDIA_TTS_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969")
TEXT = "Hello! Welcome to Spelling Bee practice. Your first word is: adventure."

if not API_KEY:
    sys.exit("NVIDIA_API_KEY not set. Export it or add to .env")

print(f"Server   : {SERVER}")
print(f"Voice ID : {VOICE_ID}")
print(f"Text     : {TEXT}")
print()

# --- Test with pipecat-ai NvidiaTTSService ---
try:
    from pipecat.services.nvidia.tts import NvidiaTTSService
    from pipecat.frames.frames import StartFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

    print("Testing NVIDIA MagpieTTS via pipecat-ai...")

    async def test_tts():
        tts = NvidiaTTSService(
            api_key=API_KEY,
            server=SERVER,
            voice_id=VOICE_ID,
            sample_rate=16000,
            model_function_map={
                "function_id": FUNCTION_ID,
                "model_name": "magpie-tts-multilingual",
            },
            params=NvidiaTTSService.InputParams(),
        )

        # Initialize the service
        await tts.start(StartFrame())

        # Generate audio
        audio_chunks = []
        async for frame in tts.run_tts(TEXT, context_id="test"):
            if isinstance(frame, TTSAudioRawFrame):
                audio_chunks.append(frame.audio)

        if audio_chunks:
            audio_bytes = b"".join(audio_chunks)
            out_file = "test_output_magpie.wav"
            
            # Write raw PCM as WAV
            import wave
            with wave.open(out_file, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit PCM
                wav.setframerate(16000)
                wav.writeframes(audio_bytes)

            print(f"Saved {len(audio_bytes)} bytes -> {out_file}")
            print(f"Play it:  aplay {out_file}  (Linux) or open {out_file} (macOS)")
        else:
            print("No audio generated")

    asyncio.run(test_tts())

except ImportError as e:
    print(f"pipecat-ai[nvidia] not installed: {e}")
    print("Install with: pip install pipecat-ai[nvidia]")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
