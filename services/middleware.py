"""
Samvad XR — Middleware (Codec Layer)
=====================================
Pure sync functions for Base64 ↔ bytes conversion.
No I/O, no async — just computation.
"""

import base64
import logging

logger = logging.getLogger("samvadxr.middleware")


def base64_to_bytes(b64_string: str) -> bytes:
    """
    Decode a Base64 string into raw bytes.

    Automatically strips the ``data:audio/wav;base64,`` data-URI header
    if present.

    Args:
        b64_string: Base64-encoded string, optionally with data-URI prefix.

    Returns:
        Raw bytes (e.g. WAV audio data).

    Raises:
        ValueError: If the input is not valid Base64.
    """
    if not b64_string:
        logger.error("base64_to_bytes called with empty string")
        raise ValueError("Invalid base64 input: empty string")

    had_header = False
    # Strip data-URI header if present
    if "," in b64_string and b64_string.startswith("data:"):
        had_header = True
        b64_string = b64_string.split(",", 1)[1]

    try:
        result = base64.b64decode(b64_string, validate=True)
        logger.info(
            "base64_to_bytes: %d chars → %d bytes%s",
            len(b64_string), len(result),
            " (data-URI header stripped)" if had_header else "",
        )
        return result
    except Exception as e:
        logger.error("base64_to_bytes FAILED: %s (input: %d chars)", e, len(b64_string))
        raise ValueError(f"Invalid base64 input: {e}") from e


def bytes_to_base64(audio_bytes: bytes) -> str:
    """
    Encode raw bytes into a plain Base64 string (no data-URI header).

    Args:
        audio_bytes: Raw bytes to encode.

    Returns:
        Clean Base64 string without any prefix.
    """
    result = base64.b64encode(audio_bytes).decode("utf-8")
    logger.info("bytes_to_base64: %d bytes → %d chars", len(audio_bytes), len(result))
    return result
