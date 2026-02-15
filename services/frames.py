# Adapted from pipecat-ai/nemotron-january-2026 (MIT License)
# Custom frame types for voice agent pipeline.

from pipecat.frames.frames import SystemFrame


class ChunkedLLMContinueGenerationFrame(SystemFrame):
    """Signal frame sent upstream by TTS when a segment completes.

    Tells the LLM service that TTS has finished processing the current chunk
    and generation can continue to the next chunk.
    """
    pass
