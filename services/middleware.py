"""
Samvad XR — Middleware (Codec Layer)
=====================================
Pure sync functions for Base64 ↔ bytes conversion.
No I/O, no async — just computation.
"""

import base64


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
        raise ValueError("Invalid base64 input: empty string")

    # Strip data-URI header if present
    if "," in b64_string and b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]

    try:
        return base64.b64decode(b64_string, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 input: {e}") from e


def bytes_to_base64(audio_bytes: bytes) -> str:
    """
    Encode raw bytes into a plain Base64 string (no data-URI header).

    Args:
        audio_bytes: Raw bytes to encode.

    Returns:
        Clean Base64 string without any prefix.
    """
    return base64.b64encode(audio_bytes).decode("utf-8")
