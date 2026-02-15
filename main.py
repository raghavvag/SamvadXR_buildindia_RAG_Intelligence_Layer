"""
Samvad XR — FastAPI Application (Dev B's Entry Point)
=====================================================
Dev B owns this file. Unity talks directly to us.
We orchestrate the full pipeline and call Dev A's
generate_vendor_response() for LLM + Neo4j in Step 7.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from models import InteractRequest, InteractResponse, NegotiationState
from services.middleware import base64_to_bytes, bytes_to_base64
from services.voice_ops import transcribe_with_sarvam, speak_with_sarvam
from services.rag_ops import initialize_knowledge_base, retrieve_context
from services.context_memory import ConversationMemory
from services.exceptions import SarvamServiceError, RAGServiceError

load_dotenv()

logger = logging.getLogger("samvadxr")

# ── Session store (in-memory, Dev B manages this) ──────────────────────
sessions: dict[str, ConversationMemory] = {}


def get_memory(session_id: str) -> ConversationMemory:
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory(session_id=session_id)
    return sessions[session_id]


# ── Mock brain (used until Dev A provides real generate_vendor_response) ─
async def _mock_generate_vendor_response(
    transcribed_text: str,
    context_block: str,
    rag_context: str,
    scene_context: dict,
    session_id: str,
) -> dict:
    """Placeholder until Dev A provides the real function."""
    return {
        "reply_text": f"[MOCK] Vendor responding to: {transcribed_text[:50]}",
        "new_mood": scene_context.get("vendor_happiness", 50),
        "new_stage": scene_context.get("negotiation_stage", "GREETING"),
        "price_offered": scene_context.get("current_price", 0),
        "vendor_happiness": scene_context.get("vendor_happiness", 50),
        "vendor_patience": scene_context.get("vendor_patience", 70),
        "vendor_mood": "neutral",
    }


# Try to import Dev A's real function; fall back to mock
try:
    from brain import generate_vendor_response  # type: ignore
except ImportError:
    generate_vendor_response = _mock_generate_vendor_response
    logger.info("Dev A's brain not found — using mock generate_vendor_response")


# ── App lifespan (startup / shutdown) ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load RAG knowledge base
    logger.info("Initializing RAG knowledge base...")
    initialize_knowledge_base()
    logger.info("RAG knowledge base ready.")
    yield
    # Shutdown: cleanup if needed
    logger.info("Shutting down Samvad XR.")


app = FastAPI(
    title="Samvad XR Intelligence Engine",
    description="VR Language Immersion — Dev B's Orchestration Layer",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Main endpoint ─────────────────────────────────────────────────────
@app.post("/api/interact", response_model=InteractResponse)
async def interact(request: InteractRequest) -> InteractResponse:
    """
    Full pipeline: Audio in → STT → Memory → RAG → LLM (Dev A) → TTS → Audio out.
    Steps 1–11 as defined in execution_plan.md.
    """
    memory = get_memory(request.session_id)

    # Step 2 — Decode audio
    try:
        audio_bytes = base64_to_bytes(request.audio_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {e}")

    # Step 3 — Speech-to-Text
    try:
        transcribed_text = await transcribe_with_sarvam(
            audio_bytes, request.language_code
        )
    except SarvamServiceError as e:
        raise HTTPException(status_code=503, detail=f"Voice recognition unavailable: {e}")

    # Silence detected — short-circuit
    if transcribed_text == "":
        return InteractResponse(
            session_id=request.session_id,
            transcribed_text="",
            agent_reply_text="Kuch bola aapne?",
            agent_audio_base64="",
            vendor_mood="confused",
        )

    # Step 4 — Store user turn
    memory.add_turn(
        role="user",
        text=transcribed_text,
        metadata={
            "held_item": (
                request.scene_context.items_in_hand[0]
                if request.scene_context.items_in_hand
                else ""
            ),
            "looked_at_item": request.scene_context.looking_at,
            "vendor_happiness": request.scene_context.vendor_happiness,
            "vendor_patience": request.scene_context.vendor_patience,
            "stage": request.scene_context.negotiation_stage,
        },
    )

    # Steps 5 & 6 — Parallel: context history + RAG
    try:
        context_block, rag_context = await asyncio.gather(
            asyncio.to_thread(memory.get_context_block),
            retrieve_context(transcribed_text, n_results=3),
        )
    except RAGServiceError:
        # Graceful degradation — continue without RAG
        context_block = memory.get_context_block()
        rag_context = ""

    # Step 7 — Call Dev A's brain (LLM + Neo4j validation)
    try:
        result = await generate_vendor_response(
            transcribed_text=transcribed_text,
            context_block=context_block,
            rag_context=rag_context,
            scene_context=request.scene_context.model_dump(),
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Step 8 — Store vendor turn
    memory.add_turn(
        role="vendor",
        text=result["reply_text"],
        metadata={
            "vendor_happiness": result.get("vendor_happiness", 50),
            "vendor_patience": result.get("vendor_patience", 70),
            "stage": result.get("new_stage", "GREETING"),
            "price": result.get("price_offered", 0),
        },
    )

    # Step 9 — Text-to-Speech
    agent_audio_base64 = ""
    try:
        tts_bytes = await speak_with_sarvam(
            result["reply_text"], request.language_code
        )
        # Step 10 — Encode audio
        agent_audio_base64 = bytes_to_base64(tts_bytes)
    except SarvamServiceError:
        # Subtitle mode — text-only response
        logger.warning("TTS unavailable, sending text-only response")

    # Step 11 — Return response to Unity
    return InteractResponse(
        session_id=request.session_id,
        transcribed_text=transcribed_text,
        agent_reply_text=result["reply_text"],
        agent_audio_base64=agent_audio_base64,
        vendor_mood=result.get("vendor_mood", "neutral"),
        negotiation_state=NegotiationState(
            item=request.scene_context.looking_at,
            quoted_price=result.get("price_offered", 0),
            vendor_happiness=result.get("vendor_happiness", 50),
            vendor_patience=result.get("vendor_patience", 70),
            stage=result.get("new_stage", "GREETING"),
            turn_count=len(memory.get_recent_turns(n=100)),
            deal_status="negotiating",
        ),
    )


# ── Run with uvicorn ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
