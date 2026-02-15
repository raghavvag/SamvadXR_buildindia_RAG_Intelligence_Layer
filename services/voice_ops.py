"""
Samvad XR — Voice Operations (Sarvam AI STT/TTS)
==================================================
Async functions that bridge to Sarvam AI for
Speech-to-Text  (saaras:v3) and Text-to-Speech (bulbul:v3).

Uses the official sarvamai AsyncSarvamAI SDK.
Auto-mocks when SARVAM_API_KEY is not set.
"""

import asyncio
import base64
import logging
import os
import struct

from sarvamai import AsyncSarvamAI
from sarvamai.core.api_error import ApiError

from services.exceptions import SarvamServiceError

logger = logging.getLogger("samvadxr.voice")


def _get_api_key() -> str:
    """Read API key at call time so load_dotenv() has a chance to run first."""
    return os.getenv("SARVAM_API_KEY", "")


# ── Language code normalizer ───────────────────────────────────────────
# Frontend may send short codes ("en", "hi") — Sarvam requires BCP-47.
_LANGUAGE_MAP: dict[str, str] = {
    "en":  "en-IN",
    "hi":  "hi-IN",
    "bn":  "bn-IN",
    "kn":  "kn-IN",
    "ml":  "ml-IN",
    "mr":  "mr-IN",
    "od":  "od-IN",
    "pa":  "pa-IN",
    "ta":  "ta-IN",
    "te":  "te-IN",
    "gu":  "gu-IN",
    "as":  "as-IN",
    "ur":  "ur-IN",
    "ne":  "ne-IN",
}


def normalize_language_code(code: str) -> str:
    """Map short codes ('en') → BCP-47 ('en-IN'). Pass-through if already valid."""
    return _LANGUAGE_MAP.get(code.lower(), code)


# Retry config
_MAX_RETRIES = 2          # Total attempts (1 original + 1 retry)
_RETRY_DELAY_S = 0.5      # 500ms backoff between retries


# ── STT ────────────────────────────────────────────────────────────────

async def transcribe_with_sarvam(audio_bytes: bytes, language_code: str) -> str:
    """
    Transcribe audio bytes to text using Sarvam AI STT (saaras:v3).

    Args:
        audio_bytes: Raw WAV audio bytes (16-bit PCM).
        language_code: "hi-IN", "en-IN", or short form "hi", "en", etc.

    Returns:
        Transcribed text in native script, or "" on silence/noise.

    Raises:
        SarvamServiceError: After retry exhaustion or API error.
    """
    lang = normalize_language_code(language_code)

    # Mock mode — no API key
    api_key = _get_api_key()
    if not api_key:
        mock_text = f"[MOCK] User said something in {lang}"
        logger.info("STT mock mode — no SARVAM_API_KEY set")
        logger.info("STT input  : %d bytes audio, lang=%s", len(audio_bytes), lang)
        logger.info("STT output : %s", mock_text)
        return mock_text

    logger.info("STT input  : %d bytes audio, lang=%s (raw=%s)", len(audio_bytes), lang, language_code)

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client = AsyncSarvamAI(api_subscription_key=api_key)
            response = await client.speech_to_text.transcribe(
                file=("audio.wav", audio_bytes, "audio/wav"),
                model="saaras:v3",
                mode="transcribe",
                language_code=lang,
            )

            transcript = (response.transcript or "").strip()
            logger.info("STT success (attempt %d): %d chars", attempt, len(transcript))
            logger.info("STT transcript: \"%s\"", transcript)
            return transcript

        except ApiError as e:
            last_error = SarvamServiceError(
                service="STT",
                status_code=e.status_code,
                detail=str(e.body)[:200] if e.body else str(e)[:200],
            )
            logger.warning(
                "STT attempt %d/%d failed: HTTP %s — %s",
                attempt, _MAX_RETRIES, e.status_code, str(e.body)[:100] if e.body else str(e)[:100],
            )

        except Exception as e:
            last_error = SarvamServiceError(
                service="STT", status_code=None, detail=str(e)[:200],
            )
            logger.warning("STT attempt %d/%d error: %s", attempt, _MAX_RETRIES, e)

        # Wait before retry (but not after the last attempt)
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_DELAY_S)

    # All retries exhausted
    raise last_error  # type: ignore[misc]


# ── TTS ────────────────────────────────────────────────────────────────

async def speak_with_sarvam(text: str, language_code: str) -> bytes:
    """
    Convert text to speech using Sarvam AI TTS (bulbul:v3).

    Args:
        text: Text to speak (native script).
        language_code: Target language code ("hi-IN" or short "hi").

    Returns:
        Raw WAV audio bytes.

    Raises:
        SarvamServiceError: After retry exhaustion or API error.
    """
    lang = normalize_language_code(language_code)

    # Mock mode — no API key
    api_key = _get_api_key()
    if not api_key:
        logger.info("TTS mock mode — no SARVAM_API_KEY set")
        logger.info("TTS input  : text=\"%s\", lang=%s", text, lang)
        logger.info("TTS output : silent WAV (mock)")
        return _generate_silent_wav()

    logger.info("TTS input  : text=\"%s\", lang=%s (raw=%s)", text, lang, language_code)

    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client = AsyncSarvamAI(api_subscription_key=api_key)
            response = await client.text_to_speech.convert(
                text=text,
                target_language_code=lang,
                speaker="anushka",
                model="bulbul:v3",
            )

            audios = response.audios or []
            if not audios or not audios[0]:
                last_error = SarvamServiceError(
                    service="TTS",
                    status_code=200,
                    detail="Empty audio in response",
                )
                logger.warning("TTS attempt %d: empty audio returned", attempt)
            else:
                # SDK returns base64-encoded audio strings
                audio_bytes = base64.b64decode(audios[0])
                logger.info("TTS success (attempt %d): %d bytes", attempt, len(audio_bytes))
                logger.info("TTS output : %d bytes WAV audio", len(audio_bytes))
                return audio_bytes

        except ApiError as e:
            last_error = SarvamServiceError(
                service="TTS",
                status_code=e.status_code,
                detail=str(e.body)[:200] if e.body else str(e)[:200],
            )
            logger.warning(
                "TTS attempt %d/%d failed: HTTP %s — %s",
                attempt, _MAX_RETRIES, e.status_code, str(e.body)[:100] if e.body else str(e)[:100],
            )

        except Exception as e:
            last_error = SarvamServiceError(
                service="TTS", status_code=None, detail=str(e)[:200],
            )
            logger.warning("TTS attempt %d/%d error: %s", attempt, _MAX_RETRIES, e)

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
