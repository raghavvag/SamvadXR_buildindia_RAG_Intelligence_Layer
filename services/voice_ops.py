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
import unicodedata

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


# Human-readable names for language mismatch messages
_LANGUAGE_NAMES: dict[str, str] = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "gu-IN": "Gujarati",
    "as-IN": "Assamese",
    "ur-IN": "Urdu",
    "ne-IN": "Nepali",
}


def get_language_name(bcp47_code: str) -> str:
    """Return human-readable name for a BCP-47 language code."""
    return _LANGUAGE_NAMES.get(bcp47_code, bcp47_code)


# ── Script-based language detection ────────────────────────────────────
# Each Indian language uses a distinct script. By checking the Unicode script
# of the transcribed text we can reliably detect language mismatches even
# when Sarvam's STT forces text into the hinted language.

_LANGUAGE_SCRIPTS: dict[str, str] = {
    "en-IN": "LATIN",
    "hi-IN": "DEVANAGARI",
    "bn-IN": "BENGALI",
    "kn-IN": "KANNADA",
    "ml-IN": "MALAYALAM",
    "mr-IN": "DEVANAGARI",   # Marathi uses Devanagari
    "ta-IN": "TAMIL",
    "te-IN": "TELUGU",
    "gu-IN": "GUJARATI",
    "pa-IN": "GURMUKHI",
    "od-IN": "ORIYA",
    "as-IN": "BENGALI",      # Assamese uses Bengali script
    "ur-IN": "ARABIC",       # Urdu uses Arabic/Nastaliq
    "ne-IN": "DEVANAGARI",   # Nepali uses Devanagari
}

# Reverse map: script name → default language code
_SCRIPT_TO_LANGUAGE: dict[str, str] = {
    "LATIN":      "en-IN",
    "DEVANAGARI": "hi-IN",
    "BENGALI":    "bn-IN",
    "KANNADA":    "kn-IN",
    "MALAYALAM":  "ml-IN",
    "TAMIL":      "ta-IN",
    "TELUGU":     "te-IN",
    "GUJARATI":   "gu-IN",
    "GURMUKHI":   "pa-IN",
    "ORIYA":      "od-IN",
    "ARABIC":     "ur-IN",
}


def detect_script_from_text(text: str) -> str:
    """
    Detect the dominant Unicode script in text.

    Returns script name like 'LATIN', 'DEVANAGARI', 'BENGALI', etc.
    Works by checking the first word of each character's Unicode name.
    """
    script_counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char, "")
            if name:
                # Unicode names start with script: "DEVANAGARI LETTER KA", "LATIN SMALL LETTER A"
                script = name.split()[0]
                script_counts[script] = script_counts.get(script, 0) + 1
        except ValueError:
            pass

    if not script_counts:
        return "UNKNOWN"
    return max(script_counts, key=script_counts.get)


async def detect_language_robust(text: str, target_language: str) -> str:
    """
    Detect the actual language spoken by the user using a two-layer approach:

    Layer 1 — Script detection (fast, free, no API call):
        Compare the Unicode script of the transcribed text against the expected
        script for the target language. This catches the common case where STT
        force-transcribes English audio into Hindi hint, but the output text
        is still in Latin characters.

        e.g. target=Hindi(Devanagari) but text="hello how are you"(Latin) → MISMATCH

    Layer 2 — Sarvam identify_language API (for same-script disambiguation):
        If the scripts match (e.g. Hindi vs Marathi, both Devanagari), fall back
        to Sarvam's language identification API.

    Args:
        text: The transcribed text from STT.
        target_language: The BCP-47 target language code from frontend.

    Returns:
        BCP-47 code of the detected language.
    """
    if not text or not text.strip():
        logger.info("Language detect: empty text, defaulting to target=%s", target_language)
        return target_language

    # ── Layer 1: Script-based detection ──
    actual_script = detect_script_from_text(text)
    expected_script = _LANGUAGE_SCRIPTS.get(target_language, "LATIN")

    logger.info("Language detect — script analysis:")
    logger.info("  text (first 80)   : \"%s\"", text[:80])
    logger.info("  actual_script     : %s", actual_script)
    logger.info("  expected_script   : %s (for target=%s)", expected_script, target_language)

    if actual_script == "UNKNOWN":
        logger.info("  result: UNKNOWN script, defaulting to target=%s", target_language)
        return target_language

    if actual_script != expected_script:
        # Script mismatch — user is NOT speaking the target language
        detected = _SCRIPT_TO_LANGUAGE.get(actual_script, "en-IN")
        logger.info("  result: SCRIPT MISMATCH → detected=%s (%s ≠ %s)",
                    detected, actual_script, expected_script)
        return detected

    # ── Layer 2: Same script — use Sarvam API for finer distinction ──
    logger.info("  result: scripts match (%s) → calling Sarvam API for confirmation", actual_script)
    return await detect_language_with_sarvam(text)


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


# ── Language Detection ─────────────────────────────────────────────────

async def detect_language_with_sarvam(text: str) -> str:
    """
    Auto-detect the language of transcribed text using Sarvam AI.

    Args:
        text: The transcribed text from STT.

    Returns:
        BCP-47 language code like "hi-IN", "kn-IN", "en-IN", etc.
        Defaults to "hi-IN" if detection fails.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.info("Language detect mock mode — no SARVAM_API_KEY set, defaulting to hi-IN")
        return "hi-IN"

    try:
        client = AsyncSarvamAI(api_subscription_key=api_key)
        response = await client.text.identify_language(input=text[:200])
        detected = response.language_code or "hi-IN"
        logger.info("Language detected: \"%s\" → %s", text[:50], detected)
        return detected
    except Exception as e:
        logger.warning("Language detection failed (%s), defaulting to hi-IN", e)
        return "hi-IN"


# ── TTS ────────────────────────────────────────────────────────────────

# Languages where Dev A's Hinglish/English output can be spoken directly
# (empty — translate for ALL target languages so TTS always matches)
_SKIP_TRANSLATE_LANGS: set[str] = set()


async def translate_with_sarvam(text: str, target_language_code: str) -> str:
    """
    Translate text into the target language using Sarvam AI.

    Dev A returns replies in English/Hinglish. We ALWAYS translate
    into the target language before TTS so the audio sounds natural
    and consistent regardless of what language Dev A replies in.

    Args:
        text: Source text (English/Hinglish from Dev A).
        target_language_code: BCP-47 code like "hi-IN", "kn-IN", "ta-IN", etc.

    Returns:
        Translated text, or original text if translation fails/not needed.
    """
    lang = normalize_language_code(target_language_code)

    # No translation needed for English/Hindi — Dev A already speaks those
    if lang in _SKIP_TRANSLATE_LANGS:
        return text

    api_key = _get_api_key()
    if not api_key:
        logger.info("Translate mock mode — no SARVAM_API_KEY set")
        return text

    logger.info("Translate  : \"%s\" → %s", text[:80], lang)

    try:
        client = AsyncSarvamAI(api_subscription_key=api_key)

        # Auto-detect source language — Dev A may reply in en/hi/Hinglish
        detect_resp = await client.text.identify_language(input=text[:200])
        source_lang = getattr(detect_resp, "language_code", "hi-IN") or "hi-IN"
        logger.info("Translate  : detected source=%s", source_lang)

        # If source already matches target, skip translation
        if source_lang == lang:
            logger.info("Translate  : source==target (%s), skipping", lang)
            return text

        response = await client.text.translate(
            input=text,
            source_language_code=source_lang,
            target_language_code=lang,
        )
        translated = response.translated_text or text
        logger.info("Translated : \"%s\" (%d chars)", translated[:80], len(translated))
        return translated

    except Exception as e:
        logger.warning("Translation failed (%s), using original text for TTS", e)
        return text


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
                speaker="priya",
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
