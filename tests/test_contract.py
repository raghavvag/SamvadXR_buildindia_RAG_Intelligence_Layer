"""
Phase 5.2 — Contract Tests
===========================
Verify every function signature, return type, and exception matches
what main.py expects. This is the "handshake test" — if it passes,
the pipeline is wired correctly.
"""

import asyncio
import inspect
import pytest

# ── 1. Import tests — verify every import main.py uses actually works ──

class TestImports:
    """All imports used in main.py must resolve without error."""

    def test_import_middleware_base64_to_bytes(self):
        from services.middleware import base64_to_bytes
        assert callable(base64_to_bytes)

    def test_import_middleware_bytes_to_base64(self):
        from services.middleware import bytes_to_base64
        assert callable(bytes_to_base64)

    def test_import_voice_transcribe(self):
        from services.voice_ops import transcribe_with_sarvam
        assert callable(transcribe_with_sarvam)

    def test_import_voice_speak(self):
        from services.voice_ops import speak_with_sarvam
        assert callable(speak_with_sarvam)

    def test_import_voice_normalize(self):
        from services.voice_ops import normalize_language_code
        assert callable(normalize_language_code)

    def test_import_rag_initialize(self):
        from services.rag_ops import initialize_knowledge_base
        assert callable(initialize_knowledge_base)

    def test_import_rag_retrieve(self):
        from services.rag_ops import retrieve_context
        assert callable(retrieve_context)

    def test_import_context_memory(self):
        from services.context_memory import ConversationMemory
        assert callable(ConversationMemory)

    def test_import_sarvam_error(self):
        from services.exceptions import SarvamServiceError
        assert issubclass(SarvamServiceError, Exception)

    def test_import_rag_error(self):
        from services.exceptions import RAGServiceError
        assert issubclass(RAGServiceError, Exception)

    def test_import_models(self):
        from models import InteractRequest, InteractResponse, NegotiationState, SceneContext
        for cls in (InteractRequest, InteractResponse, NegotiationState, SceneContext):
            assert callable(cls)


# ── 2. Signature tests — verify function parameters ───────────────────

class TestSignatures:
    """Every function must accept the arguments main.py passes."""

    def test_base64_to_bytes_accepts_string(self):
        from services.middleware import base64_to_bytes
        sig = inspect.signature(base64_to_bytes)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "base64_to_bytes must accept at least 1 arg"

    def test_bytes_to_base64_accepts_bytes(self):
        from services.middleware import bytes_to_base64
        sig = inspect.signature(bytes_to_base64)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "bytes_to_base64 must accept at least 1 arg"

    def test_transcribe_is_async(self):
        from services.voice_ops import transcribe_with_sarvam
        assert asyncio.iscoroutinefunction(transcribe_with_sarvam)

    def test_transcribe_accepts_bytes_and_lang(self):
        from services.voice_ops import transcribe_with_sarvam
        sig = inspect.signature(transcribe_with_sarvam)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, "transcribe_with_sarvam needs (audio_bytes, language_code)"

    def test_speak_is_async(self):
        from services.voice_ops import speak_with_sarvam
        assert asyncio.iscoroutinefunction(speak_with_sarvam)

    def test_speak_accepts_text_and_lang(self):
        from services.voice_ops import speak_with_sarvam
        sig = inspect.signature(speak_with_sarvam)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, "speak_with_sarvam needs (text, language_code)"

    def test_retrieve_context_is_async(self):
        from services.rag_ops import retrieve_context
        assert asyncio.iscoroutinefunction(retrieve_context)

    def test_retrieve_context_accepts_query(self):
        from services.rag_ops import retrieve_context
        sig = inspect.signature(retrieve_context)
        params = list(sig.parameters.keys())
        assert "query" in params or len(params) >= 1

    def test_normalize_language_code_is_sync(self):
        from services.voice_ops import normalize_language_code
        assert not asyncio.iscoroutinefunction(normalize_language_code)


# ── 3. Return type tests — verify outputs match what main.py expects ──

class TestReturnTypes:
    """Functions must return the types main.py relies on."""

    def test_base64_to_bytes_returns_bytes(self):
        from services.middleware import base64_to_bytes
        import base64
        sample = base64.b64encode(b"hello").decode()
        result = base64_to_bytes(sample)
        assert isinstance(result, bytes)

    def test_bytes_to_base64_returns_str(self):
        from services.middleware import bytes_to_base64
        result = bytes_to_base64(b"hello")
        assert isinstance(result, str)

    def test_normalize_returns_bcp47(self):
        from services.voice_ops import normalize_language_code
        result = normalize_language_code("en")
        assert isinstance(result, str)
        assert "-" in result  # e.g. "en-IN"

    def test_memory_add_turn_and_context(self):
        from services.context_memory import ConversationMemory
        m = ConversationMemory(session_id="contract-test")
        m.add_turn("user", "Hello", {"test": True})
        block = m.get_context_block()
        assert isinstance(block, str)
        assert "Hello" in block

    def test_memory_get_recent_turns_list(self):
        from services.context_memory import ConversationMemory
        m = ConversationMemory(session_id="contract-test-2")
        m.add_turn("user", "Test", {})
        turns = m.get_recent_turns(n=5)
        assert isinstance(turns, list)
        assert len(turns) == 1


# ── 4. Exception tests — verify error types are catchable ─────────────

class TestExceptions:
    """Custom exceptions must be catchable as expected in main.py."""

    def test_sarvam_error_catchable(self):
        from services.exceptions import SarvamServiceError
        try:
            raise SarvamServiceError(service="test", status_code=500, detail="boom")
        except SarvamServiceError as e:
            assert "boom" in str(e)

    def test_rag_error_catchable(self):
        from services.exceptions import RAGServiceError
        try:
            raise RAGServiceError("rag boom")
        except RAGServiceError as e:
            assert "rag boom" in str(e)

    def test_sarvam_error_is_exception(self):
        from services.exceptions import SarvamServiceError
        assert issubclass(SarvamServiceError, Exception)

    def test_rag_error_is_exception(self):
        from services.exceptions import RAGServiceError
        assert issubclass(RAGServiceError, Exception)

    def test_base64_raises_valueerror_on_bad_input(self):
        from services.middleware import base64_to_bytes
        with pytest.raises(ValueError):
            base64_to_bytes("!!!not-base64!!!")


# ── 5. FastAPI app tests — verify endpoints exist ─────────────────────

class TestAppRoutes:
    """Verify the FastAPI app has the expected routes."""

    def test_app_importable(self):
        from main import app
        assert app is not None

    def test_health_route_exists(self):
        from main import app
        paths = [r.path for r in app.routes]
        assert "/health" in paths

    def test_api_test_route_exists(self):
        from main import app
        paths = [r.path for r in app.routes]
        assert "/api/test" in paths

    def test_api_interact_route_exists(self):
        from main import app
        paths = [r.path for r in app.routes]
        assert "/api/interact" in paths
