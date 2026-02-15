"""
Samvad XR — FastAPI Application (Dev B's Entry Point)
=====================================================
Dev B owns this file. Unity talks directly to us.
We orchestrate the full pipeline and call Dev A's
generate_vendor_response() for LLM + Neo4j in Step 7.
"""

import asyncio
import base64
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # Must run BEFORE importing services that read env vars

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from models import InteractRequest, InteractResponse, NegotiationState
from services.middleware import base64_to_bytes, bytes_to_base64
from services.voice_ops import transcribe_with_sarvam, speak_with_sarvam, normalize_language_code
from services.rag_ops import initialize_knowledge_base, retrieve_context
from services.context_memory import ConversationMemory
from services.exceptions import SarvamServiceError, RAGServiceError

# ── Logging setup ──────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# Quiet noisy third-party loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

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

# CORS — allow all origins so ngrok / external clients can hit us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Raw body logger (runs BEFORE Pydantic validation) ──────────────────
@app.middleware("http")
async def log_raw_request(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        body = await request.body()
        logger.info("━" * 60)
        logger.info("📥 RAW INCOMING  %s %s", request.method, request.url.path)
        logger.info("   Content-Type : %s", request.headers.get("content-type", "N/A"))
        logger.info("   Body length  : %d bytes", len(body))
        # Print body (truncate audio_base64 if huge)
        try:
            body_str = body.decode("utf-8")
            parsed = json.loads(body_str)
            # Truncate audio_base64 for readability
            display = dict(parsed)
            for key in ("audio_base64", "audioData"):
                if key in display and len(str(display[key])) > 200:
                    display[key] = str(display[key])[:200] + f"...({len(str(parsed[key]))} chars total)"
            logger.info("   Body         : %s", json.dumps(display, indent=2, ensure_ascii=False))
        except Exception:
            logger.info("   Body (raw)   : %s", body[:500])
        logger.info("━" * 60)
    response = await call_next(request)
    return response


# ── Health ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Quick connectivity check."""
    return {"status": "ok", "service": "samvadxr-intelligence-engine"}


# ── Test endpoint (for friend / Dev A to hit via ngrok) ────────────────

from pydantic import BaseModel, Field
from typing import Any


class TestRequest(BaseModel):
    """Payload from frontend: base64 WAV + language/context fields."""
    audioData: str = Field(..., description="Base64-encoded mono WAV (16 kHz)")
    inputLanguage: str = Field("en", description="Source language code")
    targetLanguage: str = Field("en", description="Target language code")
    contextTag: str = Field("general", description="Scene/context tag")


@app.post("/api/test")
async def test_pipeline(req: TestRequest):
    """
    Test endpoint — accepts frontend JSON payload.
    Runs: decode → STT → mock brain → TTS → encode.

    Body:
    {
        "audioData": "<base64 mono WAV 16kHz>",
        "inputLanguage": "en",
        "targetLanguage": "en",
        "contextTag": "general"
    }
    """
    session_id = "test-session"
    language_code = normalize_language_code(req.inputLanguage)
    target_language = normalize_language_code(req.targetLanguage)

    # ── Log incoming request ──
    logger.info("─" * 60)
    logger.info("▶ REQUEST  /api/test")
    logger.info("  inputLanguage : %s → %s", req.inputLanguage, language_code)
    logger.info("  targetLanguage: %s → %s", req.targetLanguage, target_language)
    logger.info("  contextTag    : %s", req.contextTag)
    logger.info("  audioData     : %d chars", len(req.audioData))

    # Step 2 — Decode base64 → raw bytes
    try:
        audio_bytes = base64_to_bytes(req.audioData)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid audioData: {e}")

    # Step 3 — STT
    try:
        transcribed_text = await transcribe_with_sarvam(audio_bytes, language_code)
    except SarvamServiceError as e:
        raise HTTPException(status_code=503, detail=f"STT failed: {e}")

    # Step 4 — Store user turn in memory
    memory = get_memory(session_id)
    memory.add_turn(
        role="user",
        text=transcribed_text,
        metadata={
            "inputLanguage": req.inputLanguage,
            "targetLanguage": req.targetLanguage,
            "contextTag": req.contextTag,
        },
    )

    # Steps 5 & 6 — Parallel: context history + RAG retrieval
    rag_context = ""
    try:
        context_block, rag_context = await asyncio.gather(
            asyncio.to_thread(memory.get_context_block),
            retrieve_context(transcribed_text, n_results=3),
        )
    except RAGServiceError:
        context_block = memory.get_context_block()
        rag_context = ""
        logger.warning("RAG unavailable, continuing without context")

    logger.info("  context_block: %d chars", len(context_block))
    logger.info("  rag_context  : %d chars", len(rag_context))
    if rag_context:
        logger.info("  rag_preview  : %s", rag_context[:200])

    # Step 7 — Brain (mock or Dev A's real function)
    brain_result = await generate_vendor_response(
        transcribed_text=transcribed_text,
        context_block=context_block,
        rag_context=rag_context,
        scene_context={"contextTag": req.contextTag, "targetLanguage": req.targetLanguage},
        session_id=session_id,
    )

    # Step 8 — Store vendor turn in memory
    memory.add_turn(
        role="vendor",
        text=brain_result["reply_text"],
        metadata={"vendor_mood": brain_result.get("vendor_mood", "neutral")},
    )

    # Step 9 & 10 — TTS → base64
    agent_audio_b64 = ""
    try:
        tts_bytes = await speak_with_sarvam(brain_result["reply_text"], target_language)
        agent_audio_b64 = bytes_to_base64(tts_bytes)
    except SarvamServiceError:
        logger.warning("TTS unavailable, sending text-only response")

    response = {
        "session_id": session_id,
        "inputLanguage": req.inputLanguage,
        "targetLanguage": req.targetLanguage,
        "contextTag": req.contextTag,
        "transcribed_text": transcribed_text,
        "agent_reply_text": brain_result["reply_text"],
        "agent_audio_base64": agent_audio_b64,
        "vendor_mood": brain_result.get("vendor_mood", "neutral"),
        "rag_context_used": rag_context[:500] if rag_context else "",
    }

    # ── Log outgoing response ──
    logger.info("◀ RESPONSE /api/test")
    logger.info("  transcribed  : %s", transcribed_text)
    logger.info("  reply_text   : %s", brain_result["reply_text"])
    logger.info("  audio_out    : %d chars", len(agent_audio_b64))
    logger.info("  vendor_mood  : %s", brain_result.get("vendor_mood", "neutral"))
    logger.info("  rag_context  : %d chars", len(rag_context))
    logger.info("─" * 60)

    return response


# ── Main endpoint ─────────────────────────────────────────────────────
@app.post("/api/interact", response_model=InteractResponse)
async def interact(request: InteractRequest) -> InteractResponse:
    """
    Full pipeline: Audio in → STT → Memory → RAG → LLM (Dev A) → TTS → Audio out.
    Steps 1–11 as defined in execution_plan.md.
    """
    # ── Log incoming request ──
    logger.info("═" * 60)
    logger.info("▶ REQUEST  /api/interact")
    logger.info("  session_id   : %s", request.session_id)
    logger.info("  language_code: %s", request.language_code)
    logger.info("  audio_base64 : %d chars", len(request.audio_base64))
    logger.info("  scene_context: %s", json.dumps(request.scene_context.model_dump(), ensure_ascii=False))
    logger.info("═" * 60)

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
        logger.warning("RAG unavailable, continuing without context")

    logger.info("  context_block: %d chars", len(context_block))
    logger.info("  rag_context  : %d chars", len(rag_context))
    if rag_context:
        logger.info("  rag_preview  : %s", rag_context[:200])

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
    resp = InteractResponse(
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

    # ── Log outgoing response ──
    logger.info("═" * 60)
    logger.info("◀ RESPONSE /api/interact")
    logger.info("  transcribed  : %s", transcribed_text)
    logger.info("  reply_text   : %s", result["reply_text"])
    logger.info("  audio_out    : %d chars", len(agent_audio_base64))
    logger.info("  vendor_mood  : %s", result.get("vendor_mood", "neutral"))
    logger.info("  negotiation  : stage=%s, price=%s, happiness=%s",
                result.get("new_stage"), result.get("price_offered"), result.get("vendor_happiness"))
    logger.info("═" * 60)

    return resp


# ── Run with uvicorn ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
