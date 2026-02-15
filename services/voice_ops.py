"""
Samvad XR — Voice Operations (Sarvam AI STT/TTS)
==================================================
Stub — will be fully implemented in Phase 2.
"""

import os

from services.exceptions import SarvamServiceError

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")


async def transcribe_with_sarvam(audio_bytes: bytes, language_code: str) -> str:
    """Speech-to-Text via Sarvam AI. Stub returns mock data."""
    if not SARVAM_API_KEY:
        return f"[MOCK] User said something in {language_code}"
    raise NotImplementedError("Real STT implementation coming in Phase 2")


async def speak_with_sarvam(text: str, language_code: str) -> bytes:
    """Text-to-Speech via Sarvam AI. Stub returns silent WAV."""
    if not SARVAM_API_KEY:
        return _generate_silent_wav()
    raise NotImplementedError("Real TTS implementation coming in Phase 2")


def _generate_silent_wav() -> bytes:
    """Generate a valid 1-second silent WAV file (22kHz, 16-bit, mono)."""
    import struct

    sample_rate = 22050
    num_samples = sample_rate  # 1 second
    bits_per_sample = 16
    num_channels = 1
    data_size = num_samples * num_channels * (bits_per_sample // 8)
    # WAV header (44 bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,          # ChunkSize
        b"WAVE",
        b"fmt ",
        16,                      # Subchunk1Size (PCM)
        1,                       # AudioFormat (PCM)
        num_channels,
        sample_rate,
        sample_rate * num_channels * (bits_per_sample // 8),  # ByteRate
        num_channels * (bits_per_sample // 8),                # BlockAlign
        bits_per_sample,
        b"data",
        data_size,
    )
    # Silent samples (all zeros)
    return header + (b"\x00" * data_size)
