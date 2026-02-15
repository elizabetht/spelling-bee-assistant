#!/usr/bin/env python3
"""Patch nvidia-pipecat 0.3.0 for compatibility with pipecat >= 0.0.100.

Pipecat 0.0.100+ removed FrameSerializerType and changed the TTS service
_push_tts_frames signature. This script patches nvidia-pipecat in-place.
"""

import pathlib
import sys


def main():
    nvpc = pathlib.Path(sys.argv[1])

    # 1. Patch ace_websocket.py — remove FrameSerializerType references
    f = nvpc / "serializers" / "ace_websocket.py"
    s = f.read_text()
    s = s.replace(
        "from pipecat.serializers.base_serializer import (\n"
        "    FrameSerializer,\n"
        "    FrameSerializerType,\n"
        ")",
        "from pipecat.serializers.base_serializer import FrameSerializer",
    )
    s = s.replace("def type(self) -> FrameSerializerType:", "def type(self):")
    s = s.replace("return FrameSerializerType.BINARY", "return True")
    s = s.replace(
        "FrameSerializerType: Always returns BINARY type for this serializer.",
        "bool: Always True (binary serializer).",
    )
    s = s.replace(
        "type (FrameSerializerType): The serializer type, always BINARY.\n", ""
    )
    f.write_text(s)
    print(f"  Patched {f}")

    # 2. Patch riva_speech.py — fix _push_tts_frames signature for pipecat 0.0.102
    #    Base class now calls _push_tts_frames(AggregatedTextFrame, includes_inter_frame_spaces)
    #    but nvidia-pipecat expects _push_tts_frames(text: str)
    f = nvpc / "services" / "riva_speech.py"
    s = f.read_text()
    s = s.replace(
        "    async def _push_tts_frames(self, text: str):",
        "    async def _push_tts_frames(self, text_or_frame, includes_inter_frame_spaces=False, append_tts_text_to_context=True):",
    )
    s = s.replace(
        "        # Remove leading newlines only\n"
        '        text = text.lstrip("\\n")',
        '        text = text_or_frame.text if hasattr(text_or_frame, "text") else text_or_frame\n'
        "        # Remove leading newlines only\n"
        '        text = text.lstrip("\\n")',
    )
    f.write_text(s)
    print(f"  Patched {f}")


if __name__ == "__main__":
    main()
