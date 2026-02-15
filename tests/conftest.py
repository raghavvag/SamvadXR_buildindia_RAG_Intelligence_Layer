"""
Shared test fixtures for Samvad XR tests.
=========================================
Auto-patches language detection, translation, and Dev A brain
so pipeline tests work without real Sarvam/Dev A API calls.
"""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture(autouse=True)
def _mock_language_services():
    """
    Auto-mock services for all tests:

    - detect_language_robust: returns target_language so the gate always passes
    - translate_with_sarvam: pass-through (returns text unchanged)
    - add_conversation_to_rag: no-op
    - generate_vendor_response: uses dummy brain directly (no Dev A timeout)
    """
    import main

    # Reset session state so tests don't leak into each other
    main.sessions.clear()
    main._active_session_id = None

    async def _fake_detect(text, target_language):
        """Always returns target_language — language gate passes."""
        return target_language

    async def _fake_translate(text, target_lang):
        """Pass-through — returns text unchanged."""
        return text

    async def _fake_brain(session_id, transcribed_text, context_block,
                           rag_context, scene_context, **kwargs):
        """Use the dummy brain directly — no Dev A call."""
        return await main._dummy_generate_vendor_response(
            transcribed_text=transcribed_text,
            context_block=context_block,
            rag_context=rag_context,
            scene_context=scene_context,
            session_id=session_id,
        )

    with patch("main.detect_language_robust", new=AsyncMock(side_effect=_fake_detect)), \
         patch("main.translate_with_sarvam", new=AsyncMock(side_effect=_fake_translate)), \
         patch("main.add_conversation_to_rag", new=AsyncMock(return_value=None)), \
         patch("main.generate_vendor_response", new=AsyncMock(side_effect=_fake_brain)):
        yield
