"""
Tests for services/middleware.py — Base64 codec layer.
"""

import pytest
from services.middleware import base64_to_bytes, bytes_to_base64


class TestBase64ToBytes:
    """Tests for base64_to_bytes()."""

    def test_plain_base64_roundtrip(self):
        """Encode bytes → base64 → decode back, result matches original."""
        original = b"RIFF\x00\x00\x00\x00WAVEfmt "
        encoded = bytes_to_base64(original)
        decoded = base64_to_bytes(encoded)
        assert decoded == original

    def test_strips_data_uri_header(self):
        """Automatically strips 'data:audio/wav;base64,' prefix."""
        original = b"Hello, Samvad XR!"
        encoded = bytes_to_base64(original)
        with_header = f"data:audio/wav;base64,{encoded}"
        decoded = base64_to_bytes(with_header)
        assert decoded == original

    def test_strips_generic_data_uri(self):
        """Strips other data URI formats too."""
        original = b"\x00\x01\x02\x03"
        encoded = bytes_to_base64(original)
        with_header = f"data:application/octet-stream;base64,{encoded}"
        decoded = base64_to_bytes(with_header)
        assert decoded == original

    def test_empty_string_raises_value_error(self):
        """Empty string input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid base64 input"):
            base64_to_bytes("")

    def test_invalid_base64_raises_value_error(self):
        """Garbage input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid base64 input"):
            base64_to_bytes("not-valid-base64!!!")

    def test_wav_like_bytes(self):
        """Test with WAV-like byte pattern (RIFF header)."""
        # Minimal WAV-like header
        wav_bytes = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVE" + b"fmt " + b"\x10\x00\x00\x00"
        encoded = bytes_to_base64(wav_bytes)
        decoded = base64_to_bytes(encoded)
        assert decoded == wav_bytes
        assert decoded[:4] == b"RIFF"


class TestBytesToBase64:
    """Tests for bytes_to_base64()."""

    def test_returns_string(self):
        """Output is always a string."""
        result = bytes_to_base64(b"test")
        assert isinstance(result, str)

    def test_no_header_prefix(self):
        """Output is plain base64, no data:... prefix."""
        result = bytes_to_base64(b"test data")
        assert not result.startswith("data:")

    def test_empty_bytes(self):
        """Empty bytes produce empty base64 string."""
        result = bytes_to_base64(b"")
        assert result == ""

    def test_large_payload(self):
        """Handles large payloads (1 MB)."""
        big_data = b"\x00" * (1024 * 1024)  # 1 MB of zeros
        encoded = bytes_to_base64(big_data)
        decoded = base64_to_bytes(encoded)
        assert decoded == big_data
