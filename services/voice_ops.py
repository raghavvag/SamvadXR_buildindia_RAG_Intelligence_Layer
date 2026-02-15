"""
Samvad XR — Voice Operations (Sarvam AI STT/TTS)
==================================================
Async functions that bridge to Sarvam AI for
Speech-to-Text (saarika:v2) and Text-to-Speech (bulbul:v1).

Uses httpx.AsyncClient for non-blocking I/O.
Auto-mocks when SARVAM_API_KEY is not set.
"""

import asyncio
import base64
import logging
import os
import struct

import httpx

from services.exceptions import SarvamServiceError

logger = logging.getLogger("samvadxr.voice")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# Retry config
_MAX_RETRIES = 2          # Total attempts (1 original + 1 retry)
_RETRY_DELAY_S = 0.5      # 500ms backoff between retries
_REQUEST_TIMEOUT_S = 5.0   # Per-request timeout


# ── STT ────────────────────────────────────────────────────────────────

async def transcribe_with_sarvam(audio_bytes: bytes, language_code: str) -> str:
    """
    Transcribe audio bytes to text using Sarvam AI STT (saarika:v2).

    Args:
        audio_bytes: Raw WAV audio bytes (16-bit PCM).
        language_code: One of "hi-IN", "en-IN", "hi-EN", "kn-IN", "ta-IN".

    Returns:
        Transcribed text in native script, or "" on silence/noise.

    Raises:
        SarvamServiceError: After retry exhaustion or non-200 response.
    """
    # Mock mode — no API key
    if not SARVAM_API_KEY:
        logger.info("STT mock mode — no SARVAM_API_KEY set")
        return f"[MOCK] User said something in {language_code}"

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
                response = await client.post(
                    SARVAM_STT_URL,
                    headers={"api-subscription-key": SARVAM_API_KEY},
                    files={
                        "file": ("audio.wav", audio_bytes, "audio/wav"),
                    },
                    data={
                        "language_code": language_code,
                        "model": "saarika:v2",
                    },
                )

            if response.status_code == 200:
                data = response.json()
                transcript = data.get("transcript", "").strip()
                logger.info("STT success (attempt %d): %d chars", attempt, len(transcript))
                return transcript
            else:
                last_error = SarvamServiceError(
                    service="STT",
                    status_code=response.status_code,
                    detail=response.text[:200],
                )
                logger.warning(
                    "STT attempt %d/%d failed: HTTP %d",
                    attempt, _MAX_RETRIES, response.status_code,
                )

        except httpx.TimeoutException as e:
            last_error = SarvamServiceError(
                service="STT", status_code=None, detail=f"Timeout: {e}"
            )
            logger.warning("STT attempt %d/%d timed out", attempt, _MAX_RETRIES)

        except httpx.HTTPError as e:
            last_error = SarvamServiceError(
                service="STT", status_code=None, detail=str(e)
            )
            logger.warning("STT attempt %d/%d HTTP error: %s", attempt, _MAX_RETRIES, e)

        # Wait before retry (but not after the last attempt)
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_DELAY_S)

    # All retries exhausted
    raise last_error  # type: ignore[misc]


# ── TTS ────────────────────────────────────────────────────────────────

async def speak_with_sarvam(text: str, language_code: str) -> bytes:
    """
    Convert text to speech using Sarvam AI TTS (bulbul:v1).

    Args:
        text: Text to speak (native script).
        language_code: Target language code.

    Returns:
        Raw WAV audio bytes (16-bit PCM, 22kHz, mono).

    Raises:
        SarvamServiceError: After retry exhaustion or non-200 response.
    """
    # Mock mode — no API key
    if not SARVAM_API_KEY:
        logger.info("TTS mock mode — no SARVAM_API_KEY set")
        return _generate_silent_wav()

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
                response = await client.post(
                    SARVAM_TTS_URL,
                    headers={
                        "api-subscription-key": SARVAM_API_KEY,
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": [text],
                        "target_language_code": language_code,
                        "speaker": "meera",
                        "model": "bulbul:v1",
                    },
                )

            if response.status_code == 200:
                data = response.json()
                audios = data.get("audios", [])
                if not audios or not audios[0]:
                    # API returned empty audio — treat as error
                    last_error = SarvamServiceError(
                        service="TTS",
                        status_code=200,
                        detail="Empty audio in response",
                    )
                    logger.warning("TTS attempt %d: empty audio returned", attempt)
                else:
                    # Sarvam returns base64-encoded audio — decode to raw bytes
                    audio_bytes = base64.b64decode(audios[0])
                    logger.info(
                        "TTS success (attempt %d): %d bytes", attempt, len(audio_bytes)
                    )
                    return audio_bytes
            else:
                last_error = SarvamServiceError(
                    service="TTS",
                    status_code=response.status_code,
                    detail=response.text[:200],
                )
                logger.warning(
                    "TTS attempt %d/%d failed: HTTP %d",
                    attempt, _MAX_RETRIES, response.status_code,
                )

        except httpx.TimeoutException as e:
            last_error = SarvamServiceError(
                service="TTS", status_code=None, detail=f"Timeout: {e}"
            )
            logger.warning("TTS attempt %d/%d timed out", attempt, _MAX_RETRIES)

        except httpx.HTTPError as e:
            last_error = SarvamServiceError(
                service="TTS", status_code=None, detail=str(e)
            )
            logger.warning("TTS attempt %d/%d HTTP error: %s", attempt, _MAX_RETRIES, e)

        # Wait before retry (but not after the last attempt)
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_DELAY_S)

    # All retries exhausted
    raise last_error  # type: ignore[misc]


# ── Helpers ────────────────────────────────────────────────────────────

def _generate_silent_wav() -> bytes:
    """
    Generate a valid 1-second silent WAV file.
    Format: 22050 Hz, 16-bit PCM, mono.
    Used in mock mode so Unity receives a playable (silent) audio file.
    """
    sample_rate = 22050
    num_samples = sample_rate  # 1 second
    bits_per_sample = 16
    num_channels = 1
    data_size = num_samples * num_channels * (bits_per_sample // 8)

    # 44-byte WAV header
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,                                                    # Subchunk1Size (PCM)
        1,                                                     # AudioFormat (PCM)
        num_channels,
        sample_rate,
        sample_rate * num_channels * (bits_per_sample // 8),   # ByteRate
        num_channels * (bits_per_sample // 8),                 # BlockAlign
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + (b"\x00" * data_size)
