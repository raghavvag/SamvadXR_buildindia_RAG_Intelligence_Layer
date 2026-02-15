"""
Tests for services/voice_ops.py — Sarvam AI STT & TTS (SDK).
Covers mock mode, retry logic, error handling, WAV validation,
and language code normalization.
"""

import asyncio
import base64
import struct
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock

import pytest
import pytest_asyncio

from services.voice_ops import (
    transcribe_with_sarvam,
    speak_with_sarvam,
    _generate_silent_wav,
    normalize_language_code,
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


# ── Patch helpers ──────────────────────────────────────────────────────
_PATCH_NO_KEY = patch("services.voice_ops._get_api_key", return_value="")
_PATCH_REAL_KEY = patch("services.voice_ops._get_api_key", return_value="test-key-123")


# ── STT Mock Mode ─────────────────────────────────────────────────────

class TestSTTMockMode:
    """Tests for transcribe_with_sarvam in mock mode (no API key)."""

    @pytest.mark.asyncio
    async def test_mock_returns_string(self):
        with _PATCH_NO_KEY:
            result = await transcribe_with_sarvam(b"fake_audio", "hi-IN")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_mock_contains_language_code(self):
        with _PATCH_NO_KEY:
            result = await transcribe_with_sarvam(b"fake_audio", "hi-IN")
        assert "hi-IN" in result

    @pytest.mark.asyncio
    async def test_mock_contains_mock_marker(self):
        with _PATCH_NO_KEY:
            result = await transcribe_with_sarvam(b"fake_audio", "en-IN")
        assert "[MOCK]" in result

    @pytest.mark.asyncio
    async def test_mock_different_languages(self):
        """Each language code produces a distinct mock response."""
        with _PATCH_NO_KEY:
            r1 = await transcribe_with_sarvam(b"audio", "hi-IN")
            r2 = await transcribe_with_sarvam(b"audio", "kn-IN")
        assert "hi-IN" in r1
        assert "kn-IN" in r2
        assert r1 != r2


# ── TTS Mock Mode ─────────────────────────────────────────────────────

class TestTTSMockMode:
    """Tests for speak_with_sarvam in mock mode (no API key)."""

    @pytest.mark.asyncio
    async def test_mock_returns_bytes(self):
        with _PATCH_NO_KEY:
            result = await speak_with_sarvam("hello", "hi-IN")
        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_mock_returns_valid_wav(self):
        with _PATCH_NO_KEY:
            result = await speak_with_sarvam("hello", "hi-IN")
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    @pytest.mark.asyncio
    async def test_mock_wav_is_22khz(self):
        with _PATCH_NO_KEY:
            result = await speak_with_sarvam("namaste", "hi-IN")
        sample_rate = struct.unpack_from("<I", result, 24)[0]
        assert sample_rate == 22050


# ── Language code normalization ────────────────────────────────────────

class TestNormalizeLanguageCode:
    """Tests for normalize_language_code()."""

    def test_short_en_maps_to_en_in(self):
        assert normalize_language_code("en") == "en-IN"

    def test_short_hi_maps_to_hi_in(self):
        assert normalize_language_code("hi") == "hi-IN"

    def test_short_kn_maps_to_kn_in(self):
        assert normalize_language_code("kn") == "kn-IN"

    def test_short_ta_maps_to_ta_in(self):
        assert normalize_language_code("ta") == "ta-IN"

    def test_already_bcp47_passes_through(self):
        assert normalize_language_code("hi-IN") == "hi-IN"

    def test_unknown_code_passes_through(self):
        assert normalize_language_code("fr-FR") == "fr-FR"

    def test_case_insensitive(self):
        assert normalize_language_code("EN") == "en-IN"
        assert normalize_language_code("Hi") == "hi-IN"


# ── STT Real API Path (mocked SDK) ────────────────────────────────────

def _mock_stt_response(transcript: str = "भाई ये कितने का है?"):
    """Create a mock SpeechToTextResponse."""
    resp = MagicMock()
    resp.transcript = transcript
    resp.language_code = "hi-IN"
    resp.request_id = "test-req-123"
    return resp


def _mock_tts_response(audio_b64: str | None = None):
    """Create a mock TextToSpeechResponse."""
    if audio_b64 is None:
        audio_b64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEtest_audio_data").decode()
    resp = MagicMock()
    resp.audios = [audio_b64]
    resp.request_id = "test-req-456"
    return resp


def _patch_sdk_stt(mock_transcribe):
    """Patch AsyncSarvamAI to return a mock client with STT."""
    mock_client = MagicMock()
    mock_client.speech_to_text.transcribe = mock_transcribe
    return patch("services.voice_ops.AsyncSarvamAI", return_value=mock_client)


def _patch_sdk_tts(mock_convert):
    """Patch AsyncSarvamAI to return a mock client with TTS."""
    mock_client = MagicMock()
    mock_client.text_to_speech.convert = mock_convert
    return patch("services.voice_ops.AsyncSarvamAI", return_value=mock_client)


class TestSTTRealAPI:
    """Tests for STT with mocked SDK calls (simulating Sarvam API)."""

    @pytest.mark.asyncio
    async def test_success_returns_transcript(self):
        mock_transcribe = AsyncMock(return_value=_mock_stt_response("भाई ये कितने का है?"))
        with _PATCH_REAL_KEY, _patch_sdk_stt(mock_transcribe):
            result = await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert result == "भाई ये कितने का है?"

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_empty_string(self):
        mock_transcribe = AsyncMock(return_value=_mock_stt_response(""))
        with _PATCH_REAL_KEY, _patch_sdk_stt(mock_transcribe):
            result = await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert result == ""

    @pytest.mark.asyncio
    async def test_none_transcript_returns_empty_string(self):
        resp = MagicMock()
        resp.transcript = None
        mock_transcribe = AsyncMock(return_value=resp)
        with _PATCH_REAL_KEY, _patch_sdk_stt(mock_transcribe):
            result = await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert result == ""

    @pytest.mark.asyncio
    async def test_normalizes_short_language_code(self):
        """Passing 'en' should result in SDK call with 'en-IN'."""
        mock_transcribe = AsyncMock(return_value=_mock_stt_response("hello"))
        with _PATCH_REAL_KEY, _patch_sdk_stt(mock_transcribe):
            result = await transcribe_with_sarvam(_make_wav_bytes(), "en")
        assert result == "hello"
        # Verify the SDK was called with normalized code
        call_kwargs = mock_transcribe.call_args.kwargs
        assert call_kwargs["language_code"] == "en-IN"

    @pytest.mark.asyncio
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_retry_on_api_error_then_success(self):
        """First call raises ApiError, second succeeds."""
        from sarvamai.core.api_error import ApiError
        mock_transcribe = AsyncMock(
            side_effect=[
                ApiError(status_code=500, body={"error": "internal"}),
                _mock_stt_response("retry worked"),
            ]
        )
        with _PATCH_REAL_KEY, _patch_sdk_stt(mock_transcribe):
            result = await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert result == "retry worked"
        assert mock_transcribe.call_count == 2

    @pytest.mark.asyncio
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_after_all_retries_exhausted(self):
        """Both attempts fail → SarvamServiceError raised."""
        from sarvamai.core.api_error import ApiError
        mock_transcribe = AsyncMock(
            side_effect=ApiError(status_code=503, body={"error": "unavailable"})
        )
        with _PATCH_REAL_KEY, _patch_sdk_stt(mock_transcribe):
            with pytest.raises(SarvamServiceError) as exc_info:
                await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert exc_info.value.service == "STT"
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_on_generic_exception(self):
        """Generic exception → SarvamServiceError."""
        mock_transcribe = AsyncMock(side_effect=ConnectionError("connection lost"))
        with _PATCH_REAL_KEY, _patch_sdk_stt(mock_transcribe):
            with pytest.raises(SarvamServiceError) as exc_info:
                await transcribe_with_sarvam(_make_wav_bytes(), "hi-IN")
        assert exc_info.value.service == "STT"
        assert "connection lost" in exc_info.value.detail


# ── TTS Real API Path (mocked SDK) ────────────────────────────────────

class TestTTSRealAPI:
    """Tests for TTS with mocked SDK calls (simulating Sarvam API)."""

    @pytest.mark.asyncio
    async def test_success_returns_decoded_bytes(self):
        fake_audio = b"RIFF\x00\x00\x00\x00WAVEtest_audio_data"
        fake_b64 = base64.b64encode(fake_audio).decode()
        mock_convert = AsyncMock(return_value=_mock_tts_response(fake_b64))
        with _PATCH_REAL_KEY, _patch_sdk_tts(mock_convert):
            result = await speak_with_sarvam("नमस्ते", "hi-IN")
        assert result == fake_audio

    @pytest.mark.asyncio
    async def test_normalizes_short_language_code(self):
        """Passing 'hi' should result in SDK call with 'hi-IN'."""
        mock_convert = AsyncMock(return_value=_mock_tts_response())
        with _PATCH_REAL_KEY, _patch_sdk_tts(mock_convert):
            await speak_with_sarvam("test", "hi")
        call_kwargs = mock_convert.call_args.kwargs
        assert call_kwargs["target_language_code"] == "hi-IN"

    @pytest.mark.asyncio
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_retry_on_api_error_then_success(self):
        from sarvamai.core.api_error import ApiError
        fake_audio = b"RIFF_audio"
        fake_b64 = base64.b64encode(fake_audio).decode()
        mock_convert = AsyncMock(
            side_effect=[
                ApiError(status_code=500, body={"error": "internal"}),
                _mock_tts_response(fake_b64),
            ]
        )
        with _PATCH_REAL_KEY, _patch_sdk_tts(mock_convert):
            result = await speak_with_sarvam("test", "hi-IN")
        assert result == fake_audio
        assert mock_convert.call_count == 2

    @pytest.mark.asyncio
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_after_all_retries_exhausted(self):
        from sarvamai.core.api_error import ApiError
        mock_convert = AsyncMock(
            side_effect=ApiError(status_code=503, body={"error": "unavailable"})
        )
        with _PATCH_REAL_KEY, _patch_sdk_tts(mock_convert):
            with pytest.raises(SarvamServiceError) as exc_info:
                await speak_with_sarvam("test", "hi-IN")
        assert exc_info.value.service == "TTS"
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_on_empty_audio_response(self):
        """API returns response but empty audios → error after retries."""
        resp = MagicMock()
        resp.audios = []
        resp.request_id = "test"
        mock_convert = AsyncMock(return_value=resp)
        with _PATCH_REAL_KEY, _patch_sdk_tts(mock_convert):
            with pytest.raises(SarvamServiceError) as exc_info:
                await speak_with_sarvam("test", "hi-IN")
        assert exc_info.value.service == "TTS"

    @pytest.mark.asyncio
    @patch("services.voice_ops._RETRY_DELAY_S", 0.01)
    async def test_raises_on_generic_exception(self):
        mock_convert = AsyncMock(side_effect=ConnectionError("timeout"))
        with _PATCH_REAL_KEY, _patch_sdk_tts(mock_convert):
            with pytest.raises(SarvamServiceError) as exc_info:
                await speak_with_sarvam("test", "hi-IN")
        assert exc_info.value.service == "TTS"
        assert "timeout" in exc_info.value.detail


# ── Integration: middleware + voice roundtrip ──────────────────────────

class TestMiddlewareVoiceRoundtrip:
    """Step 2.4 — Full roundtrip in mock mode: base64 → STT → TTS → base64."""

    @pytest.mark.asyncio
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

        with _PATCH_NO_KEY:
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
