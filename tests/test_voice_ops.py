"""
Tests for services/voice_ops.py — Sarvam AI STT & TTS.
Covers mock mode, retry logic, error handling, and WAV validation.
"""

import asyncio
import base64
import struct
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest
import pytest_asyncio

from services.voice_ops import (
    transcribe_with_sarvam,
    speak_with_sarvam,
    _generate_silent_wav,
)
from services.exceptions import SarvamServiceError


# ── Helper ─────────────────────────────────────────────────────────────

def _make_wav_bytes(duration_s: float = 0.1) -> bytes:
    """Create a tiny valid WAV file for test input."""
    sample_rate = 22050
    num_samples = int(sample_rate * duration_s)
    bits_per_sample = 16
    num_channels = 1
    data_size = num_samples * num_channels * (bits_per_sample // 8)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16, 1,
        num_channels, sample_rate,
        sample_rate * num_channels * (bits_per_sample // 8),
        num_channels * (bits_per_sample // 8),
        bits_per_sample, b"data", data_size,
    )
    return header + (b"\x00" * data_size)


# ── Silent WAV generator ──────────────────────────────────────────────

class TestGenerateSilentWav:
    """Tests for _generate_silent_wav() helper."""

    def test_returns_bytes(self):
        wav = _generate_silent_wav()
        assert isinstance(wav, bytes)

    def test_has_riff_header(self):
        wav = _generate_silent_wav()
        assert wav[:4] == b"RIFF"

    def test_has_wave_format(self):
        wav = _generate_silent_wav()
        assert wav[8:12] == b"WAVE"

    def test_correct_sample_rate(self):
        wav = _generate_silent_wav()
        # Sample rate is at bytes 24-28 (little-endian uint32)
        sample_rate = struct.unpack_from("<I", wav, 24)[0]
        assert sample_rate == 22050

    def test_correct_bit_depth(self):
        wav = _generate_silent_wav()
        # Bits per sample at bytes 34-36 (little-endian uint16)
        bps = struct.unpack_from("<H", wav, 34)[0]
        assert bps == 16

    def test_one_second_duration(self):
        wav = _generate_silent_wav()
        # data_size = 22050 samples × 1 channel × 2 bytes = 44100
        expected_data_size = 22050 * 1 * 2
        # Total WAV = 44 header + data_size
        assert len(wav) == 44 + expected_data_size

    def test_data_is_silent(self):
        wav = _generate_silent_wav()
        data = wav[44:]  # Skip header
        assert data == b"\x00" * len(data)


# ── STT Mock Mode ─────────────────────────────────────────────────────

class TestSTTMockMode:
    """Tests for transcribe_with_sarvam in mock mode (no API key)."""

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "")
    async def test_mock_returns_string(self):
        result = await transcribe_with_sarvam(b"fake_audio", "hi-IN")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "")
    async def test_mock_contains_language_code(self):
        result = await transcribe_with_sarvam(b"fake_audio", "hi-IN")
        assert "hi-IN" in result

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "")
    async def test_mock_contains_mock_marker(self):
        result = await transcribe_with_sarvam(b"fake_audio", "en-IN")
        assert "[MOCK]" in result

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "")
    async def test_mock_different_languages(self):
        """Each language code produces a distinct mock response."""
        r1 = await transcribe_with_sarvam(b"audio", "hi-IN")
        r2 = await transcribe_with_sarvam(b"audio", "kn-IN")
        assert "hi-IN" in r1
        assert "kn-IN" in r2
        assert r1 != r2


# ── TTS Mock Mode ─────────────────────────────────────────────────────

class TestTTSMockMode:
    """Tests for speak_with_sarvam in mock mode (no API key)."""

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "")
    async def test_mock_returns_bytes(self):
        result = await speak_with_sarvam("hello", "hi-IN")
        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "")
    async def test_mock_returns_valid_wav(self):
        result = await speak_with_sarvam("hello", "hi-IN")
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "")
    async def test_mock_wav_is_22khz(self):
        result = await speak_with_sarvam("namaste", "hi-IN")
        sample_rate = struct.unpack_from("<I", result, 24)[0]
        assert sample_rate == 22050


# ── STT Real API Path (mocked httpx) ──────────────────────────────────

class TestSTTRealAPI:
    """Tests for STT with mocked httpx calls (simulating Sarvam API)."""

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    async def test_success_returns_transcript(self):
        mock_response = httpx.Response(
            200,
            json={"transcript": "भाई ये कितने का है?"},
            request=httpx.Request("POST", "https://api.sarvam.ai/speech-to-text"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert result == "भाई ये कितने का है?"

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    async def test_empty_transcript_returns_empty_string(self):
        mock_response = httpx.Response(
            200,
            json={"transcript": ""},
            request=httpx.Request("POST", "https://api.sarvam.ai/speech-to-text"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert result == ""

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)  # Speed up test
    async def test_retry_on_failure_then_success(self):
        """First call fails (500), second succeeds."""
        fail_response = httpx.Response(
            500,
            text="Internal Server Error",
            request=httpx.Request("POST", "https://api.sarvam.ai/speech-to-text"),
        )
        success_response = httpx.Response(
            200,
            json={"transcript": "retry worked"},
            request=httpx.Request("POST", "https://api.sarvam.ai/speech-to-text"),
        )
        mock_post = AsyncMock(side_effect=[fail_response, success_response])
        with patch("httpx.AsyncClient.post", mock_post):
            result = await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert result == "retry worked"
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_after_all_retries_exhausted(self):
        """Both attempts fail → SarvamServiceError raised."""
        fail_response = httpx.Response(
            503,
            text="Service Unavailable",
            request=httpx.Request("POST", "https://api.sarvam.ai/speech-to-text"),
        )
        mock_post = AsyncMock(return_value=fail_response)
        with patch("httpx.AsyncClient.post", mock_post):
            with pytest.raises(SarvamServiceError) as exc_info:
                await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert exc_info.value.service == "STT"
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_on_timeout(self):
        """Timeout on both attempts → SarvamServiceError."""
        mock_post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        with patch("httpx.AsyncClient.post", mock_post):
            with pytest.raises(SarvamServiceError) as exc_info:
                await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert exc_info.value.service == "STT"
        assert "Timeout" in exc_info.value.detail


# ── TTS Real API Path (mocked httpx) ──────────────────────────────────

class TestTTSRealAPI:
    """Tests for TTS with mocked httpx calls (simulating Sarvam API)."""

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    async def test_success_returns_decoded_bytes(self):
        fake_audio = b"RIFF\x00\x00\x00\x00WAVEtest_audio_data"
        fake_b64 = base64.b64encode(fake_audio).decode()
        mock_response = httpx.Response(
            200,
            json={"audios": [fake_b64]},
            request=httpx.Request("POST", "https://api.sarvam.ai/text-to-speech"),
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await speak_with_sarvam("नमस्ते", "hi-IN")
        assert result == fake_audio

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_retry_on_failure_then_success(self):
        fake_audio = b"RIFF_audio"
        fake_b64 = base64.b64encode(fake_audio).decode()
        fail_response = httpx.Response(
            500,
            text="Error",
            request=httpx.Request("POST", "https://api.sarvam.ai/text-to-speech"),
        )
        success_response = httpx.Response(
            200,
            json={"audios": [fake_b64]},
            request=httpx.Request("POST", "https://api.sarvam.ai/text-to-speech"),
        )
        mock_post = AsyncMock(side_effect=[fail_response, success_response])
        with patch("httpx.AsyncClient.post", mock_post):
            result = await speak_with_sarvam("test", "hi-IN")
        assert result == fake_audio
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_after_all_retries_exhausted(self):
        fail_response = httpx.Response(
            503,
            text="Down",
            request=httpx.Request("POST", "https://api.sarvam.ai/text-to-speech"),
        )
        mock_post = AsyncMock(return_value=fail_response)
        with patch("httpx.AsyncClient.post", mock_post):
            with pytest.raises(SarvamServiceError) as exc_info:
                await speak_with_sarvam("test", "hi-IN")
        assert exc_info.value.service == "TTS"
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_on_empty_audio_response(self):
        """API returns 200 but empty audios array → error after retries."""
        mock_response = httpx.Response(
            200,
            json={"audios": []},
            request=httpx.Request("POST", "https://api.sarvam.ai/text-to-speech"),
        )
        mock_post = AsyncMock(return_value=mock_response)
        with patch("httpx.AsyncClient.post", mock_post):
            with pytest.raises(SarvamServiceError) as exc_info:
                await speak_with_sarvam("test", "hi-IN")
        assert exc_info.value.service == "TTS"

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "test-key-123")
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_on_timeout(self):
        mock_post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        with patch("httpx.AsyncClient.post", mock_post):
            with pytest.raises(SarvamServiceError) as exc_info:
                await speak_with_sarvam("test", "hi-IN")
        assert exc_info.value.service == "TTS"
        assert "Timeout" in exc_info.value.detail


# ── Integration: middleware + voice roundtrip ──────────────────────────

class TestMiddlewareVoiceRoundtrip:
    """Step 2.4 — Full roundtrip in mock mode: base64 → STT → TTS → base64."""

    @pytest.mark.asyncio
    @patch("services.voice_ops.SARVAM_API_KEY", "")
    async def test_full_roundtrip_mock_mode(self):
        """
        Simulate the full audio pipeline in mock mode:
        1. base64 → bytes (decode input audio)
        2. bytes → STT → text (mock transcription)
        3. text → TTS → bytes (mock silent WAV)
        4. bytes → base64 (encode output audio)
        """
        from services.middleware import base64_to_bytes, bytes_to_base64

        # Step 1: Create mock input audio (base64-encoded WAV)
        input_wav = _make_wav_bytes(0.5)
        input_b64 = bytes_to_base64(input_wav)

        # Step 2: Decode
        decoded = base64_to_bytes(input_b64)
        assert decoded == input_wav

        # Step 3: STT (mock)
        text = await transcribe_with_sarvam(decoded, "hi-IN")
        assert isinstance(text, str)
        assert len(text) > 0

        # Step 4: TTS (mock → silent WAV)
        tts_bytes = await speak_with_sarvam(text, "hi-IN")
        assert isinstance(tts_bytes, bytes)
        assert tts_bytes[:4] == b"RIFF"

        # Step 5: Encode for Unity
        output_b64 = bytes_to_base64(tts_bytes)
        assert isinstance(output_b64, str)
        assert len(output_b64) > 0

        # Verify we can decode it back
        round_tripped = base64_to_bytes(output_b64)
        assert round_tripped == tts_bytes
