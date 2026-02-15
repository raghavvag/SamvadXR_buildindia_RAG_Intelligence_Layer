"""
Phase 5.4 — Full Pipeline Smoke Tests
=======================================
End-to-end tests through the FastAPI endpoint with mocked Sarvam APIs.
Tests the COMPLETE 11-step pipeline: audio in → STT → memory → RAG →
brain → memory → TTS → audio out.

All Sarvam API calls are mocked so these run without an API key.
The dummy brain in main.py is used (no Dev A dependency).
"""

import asyncio
import base64
import json
import struct
import wave
import io

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from main import app


# ── Helpers ────────────────────────────────────────────────────────────

def _make_wav_base64(duration_s: float = 0.5, sample_rate: int = 16000) -> str:
    """Generate a valid WAV file encoded as base64."""
    n_samples = int(sample_rate * duration_s)
    # Sine-ish samples (just non-zero data)
    samples = b"\x00\x01" * n_samples
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples)
    return base64.b64encode(buf.getvalue()).decode()


VALID_WAV_B64 = _make_wav_base64(0.5)

# Patch targets — must patch where used (main module), not where defined
_PATCH_STT = "main.transcribe_with_sarvam"
_PATCH_TTS = "main.speak_with_sarvam"


# ── /api/test — Full Pipeline Tests ───────────────────────────────────

class TestPipelineEndToEnd:
    """Full 11-step pipeline through /api/test endpoint."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_all_fields(self):
        """Complete request → response with all 4 expected fields."""
        mock_stt = AsyncMock(return_value="Bhaiya ye kitne ka hai?")
        mock_tts = AsyncMock(return_value=b"fake-tts-audio-bytes")

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "hi",
                    "targetLanguage": "hi",
                    "object_grabbed": "silk_scarf",
                    "happiness_score": 50,
                    "negotiation_state": "GREETING",
                })

        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert "audioReply" in data
        assert "negotiation_state" in data
        assert "happiness_score" in data

    @pytest.mark.asyncio
    async def test_reply_contains_vendor_text(self):
        """reply field contains the vendor's response text."""
        mock_stt = AsyncMock(return_value="Testing transcription")
        mock_tts = AsyncMock(return_value=b"audio")

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "en",
                    "targetLanguage": "en",
                    "object_grabbed": "Tomato",
                    "happiness_score": 60,
                    "negotiation_state": "GREETING",
                })

        # reply is now the vendor's text, not user transcription
        data = resp.json()
        assert len(data["reply"]) > 0
        assert data["reply"] != "Testing transcription"  # NOT user text

    @pytest.mark.asyncio
    async def test_tts_audio_returned_as_base64(self):
        """audioReply is a non-empty base64 string when TTS succeeds."""
        mock_stt = AsyncMock(return_value="Hello vendor")
        mock_tts = AsyncMock(return_value=b"\x00\x01\x02\x03" * 100)

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "en",
                    "targetLanguage": "en",
                    "object_grabbed": "",
                    "happiness_score": 50,
                    "negotiation_state": "GREETING",
                })

        data = resp.json()
        assert len(data["audioReply"]) > 0
        # Verify it's valid base64
        decoded = base64.b64decode(data["audioReply"])
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_negotiation_state_advances(self):
        """Dummy brain advances GREETING → INQUIRY."""
        mock_stt = AsyncMock(return_value="Namaste bhaiya")
        mock_tts = AsyncMock(return_value=b"audio")

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "hi",
                    "targetLanguage": "hi",
                    "object_grabbed": "Mango",
                    "happiness_score": 50,
                    "negotiation_state": "GREETING",
                })

        data = resp.json()
        assert data["negotiation_state"] == "INQUIRY"

    @pytest.mark.asyncio
    async def test_happiness_score_returned(self):
        """Response includes updated happiness_score from brain."""
        mock_stt = AsyncMock(return_value="Kitne ka hai?")
        mock_tts = AsyncMock(return_value=b"audio")

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "hi",
                    "targetLanguage": "hi",
                    "object_grabbed": "Tomato",
                    "happiness_score": 50,
                    "negotiation_state": "GREETING",
                })

        data = resp.json()
        assert isinstance(data["happiness_score"], int)

    @pytest.mark.asyncio
    async def test_silence_returns_empty_with_happiness(self):
        """Empty transcription → silence response with original happiness_score."""
        mock_stt = AsyncMock(return_value="")

        with patch(_PATCH_STT, mock_stt):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "en",
                    "targetLanguage": "en",
                    "object_grabbed": "",
                    "happiness_score": 75,
                    "negotiation_state": "BARGAINING",
                })

        data = resp.json()
        assert data["reply"] == ""
        assert data["audioReply"] == ""
        assert data["negotiation_state"] == "BARGAINING"
        assert data["happiness_score"] == 75

    @pytest.mark.asyncio
    async def test_tts_failure_returns_text_only(self):
        """When TTS fails, audioReply is empty but text response works."""
        from services.exceptions import SarvamServiceError

        mock_stt = AsyncMock(return_value="Test speech")
        mock_tts = AsyncMock(side_effect=SarvamServiceError(
            service="TTS", status_code=500, detail="Service down"
        ))

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "en",
                    "targetLanguage": "en",
                    "object_grabbed": "",
                    "happiness_score": 50,
                    "negotiation_state": "GREETING",
                })

        assert resp.status_code == 200
        data = resp.json()
        # reply is vendor text, not user input
        assert len(data["reply"]) > 0
        assert data["audioReply"] == ""
        assert "negotiation_state" in data
        assert "happiness_score" in data

    @pytest.mark.asyncio
    async def test_stt_failure_returns_503(self):
        """When STT fails, we return 503."""
        from services.exceptions import SarvamServiceError

        mock_stt = AsyncMock(side_effect=SarvamServiceError(
            service="STT", status_code=500, detail="Service down"
        ))

        with patch(_PATCH_STT, mock_stt):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "en",
                    "targetLanguage": "en",
                    "object_grabbed": "",
                    "happiness_score": 50,
                    "negotiation_state": "GREETING",
                })

        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_400(self):
        """Bad audioData → 400 error."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/test", json={
                "audioData": "!!!totally-not-base64!!!",
                "inputLanguage": "en",
                "targetLanguage": "en",
                "object_grabbed": "",
                "happiness_score": 50,
                "negotiation_state": "GREETING",
            })

        assert resp.status_code == 400


# ── Full Negotiation Flow (GREETING → DEAL_CLOSED) ───────────────────

class TestFullNegotiationFlow:
    """Simulate a complete 5-turn negotiation through the dummy brain."""

    @pytest.mark.asyncio
    async def test_full_negotiation_greeting_to_deal(self):
        """Walk through all 5 states: GREETING → INQUIRY → BARGAINING → COUNTER → DEAL_CLOSED."""
        mock_stt = AsyncMock(return_value="Bhaiya baat karo")
        mock_tts = AsyncMock(return_value=b"tts-audio")

        states = ["GREETING", "INQUIRY", "BARGAINING", "COUNTER", "DEAL_CLOSED"]
        expected_next = ["INQUIRY", "BARGAINING", "COUNTER", "DEAL_CLOSED", "DEAL_CLOSED"]

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                for i, (current, expected) in enumerate(zip(states, expected_next)):
                    resp = await ac.post("/api/test", json={
                        "audioData": VALID_WAV_B64,
                        "inputLanguage": "hi",
                        "targetLanguage": "hi",
                        "object_grabbed": "Silk_Scarf",
                        "happiness_score": 50,
                        "negotiation_state": current,
                    })

                    assert resp.status_code == 200, f"Turn {i+1} failed with {resp.status_code}"
                    data = resp.json()
                    assert data["negotiation_state"] == expected, \
                        f"Turn {i+1}: expected state '{expected}', got '{data['negotiation_state']}'"
                    assert isinstance(data["happiness_score"], int), \
                        f"Turn {i+1}: happiness_score not int"
                    assert len(data["reply"]) > 0, f"Turn {i+1}: empty reply"

    @pytest.mark.asyncio
    async def test_happiness_changes_through_negotiation(self):
        """Happiness should decrease during BARGAINING and increase on DEAL_CLOSED."""
        mock_stt = AsyncMock(return_value="Let's negotiate")
        mock_tts = AsyncMock(return_value=b"audio")

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                # BARGAINING → happiness should drop
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "en",
                    "targetLanguage": "en",
                    "object_grabbed": "Vase",
                    "happiness_score": 60,
                    "negotiation_state": "INQUIRY",
                })
                data = resp.json()
                assert data["happiness_score"] < 60, "Happiness should decrease during BARGAINING"

                # COUNTER → DEAL_CLOSED, happiness should rise
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "en",
                    "targetLanguage": "en",
                    "object_grabbed": "Vase",
                    "happiness_score": 40,
                    "negotiation_state": "COUNTER",
                })
                data = resp.json()
                assert data["happiness_score"] > 40, "Happiness should increase on DEAL_CLOSED"


# ── /health endpoint ──────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Language normalization through endpoint ───────────────────────────

class TestLanguageNormalization:
    """Verify short language codes work end-to-end."""

    @pytest.mark.asyncio
    async def test_short_codes_work(self):
        mock_stt = AsyncMock(return_value="Test")
        mock_tts = AsyncMock(return_value=b"audio")

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                for code in ["en", "hi", "kn", "ta", "te"]:
                    resp = await ac.post("/api/test", json={
                        "audioData": VALID_WAV_B64,
                        "inputLanguage": code,
                        "targetLanguage": code,
                        "object_grabbed": "",
                        "happiness_score": 50,
                        "negotiation_state": "GREETING",
                    })
                    assert resp.status_code == 200, f"Language '{code}' failed"


# ── Memory persistence across requests ────────────────────────────────

class TestMemoryPersistence:
    """Memory should accumulate turns within a session."""

    @pytest.mark.asyncio
    async def test_memory_stores_across_turns(self):
        mock_stt = AsyncMock(side_effect=["Turn one", "Turn two", "Turn three"])
        mock_tts = AsyncMock(return_value=b"audio")

        # First request with GREETING creates a new session,
        # subsequent requests with INQUIRY/BARGAINING continue it
        states = ["GREETING", "INQUIRY", "BARGAINING"]

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                for i, state in enumerate(states):
                    resp = await ac.post("/api/test", json={
                        "audioData": VALID_WAV_B64,
                        "inputLanguage": "en",
                        "targetLanguage": "en",
                        "object_grabbed": "Apple",
                        "happiness_score": 50,
                        "negotiation_state": state,
                    })
                    assert resp.status_code == 200

        # Verify memory has stored turns (user + vendor for each = 6 turns)
        from main import _active_session_id, sessions
        assert _active_session_id is not None
        mem = sessions[_active_session_id]
        turns = mem.get_recent_turns(n=100)
        assert len(turns) >= 6, f"Expected at least 6 turns, got {len(turns)}"


# ── RAG integration ───────────────────────────────────────────────────

class TestRAGIntegration:
    """RAG context should be retrieved during pipeline execution."""

    @pytest.mark.asyncio
    async def test_rag_retrieves_context(self):
        """Pipeline completes even when RAG returns results."""
        mock_stt = AsyncMock(return_value="Namaste bhaiya silk scarf dikhao")
        mock_tts = AsyncMock(return_value=b"audio")

        with patch(_PATCH_STT, mock_stt), patch(_PATCH_TTS, mock_tts):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "hi",
                    "targetLanguage": "hi",
                    "object_grabbed": "silk_scarf",
                    "happiness_score": 50,
                    "negotiation_state": "GREETING",
                })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["reply"]) > 0

    @pytest.mark.asyncio
    async def test_rag_failure_graceful_degradation(self):
        """Pipeline works even if RAG fails."""
        from services.exceptions import RAGServiceError

        mock_stt = AsyncMock(return_value="Hello")
        mock_tts = AsyncMock(return_value=b"audio")
        mock_rag = AsyncMock(side_effect=RAGServiceError("ChromaDB down"))

        with patch(_PATCH_STT, mock_stt), \
             patch(_PATCH_TTS, mock_tts), \
             patch("main.retrieve_context", mock_rag):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test", json={
                    "audioData": VALID_WAV_B64,
                    "inputLanguage": "en",
                    "targetLanguage": "en",
                    "object_grabbed": "",
                    "happiness_score": 50,
                    "negotiation_state": "GREETING",
                })

        assert resp.status_code == 200
        data = resp.json()
        # reply is vendor text (not "Hello")
        assert len(data["reply"]) > 0
        assert "negotiation_state" in data
        assert "happiness_score" in data
