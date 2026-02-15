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
import time
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


# ── Dummy brain for testing (skip Dev A for now) ──────────────────────
_DUMMY_REPLIES = {
    "GREETING":    "Arre sahab! Aao aao, dekhiye humari dukaan mein bahut achhi cheezein hain!",
    "INQUIRY":     "Yeh bilkul asli hai sahab, pure silk ka hai. Aap chhukar dekhiye!",
    "BARGAINING":  "Itna kam? Sahab yeh toh cost price se bhi kam hai! Chaliye thoda aur badhaaiye.",
    "COUNTER":     "Achha chaliye aapke liye special price — 800 rupaye, final offer!",
    "DEAL_CLOSED": "Bahut achha! Deal pakki. Aapne achhi kharidari ki sahab!",
}

_STATE_TRANSITIONS = {
    "GREETING":    "INQUIRY",
    "INQUIRY":     "BARGAINING",
    "BARGAINING":  "COUNTER",
    "COUNTER":     "DEAL_CLOSED",
    "DEAL_CLOSED": "DEAL_CLOSED",
}


async def _dummy_generate_vendor_response(
    transcribed_text: str,
    context_block: str,
    rag_context: str,
    scene_context: dict,
    session_id: str,
) -> dict:
    """Hardcoded dummy response for end-to-end testing without Dev A's brain."""
    current_state = scene_context.get("negotiation_state", "GREETING")
    next_state = _STATE_TRANSITIONS.get(current_state, "GREETING")
    happiness = scene_context.get("happiness_score", 50)

    # Slightly adjust happiness based on state
    if next_state == "BARGAINING":
        happiness = max(0, happiness - 10)
    elif next_state == "DEAL_CLOSED":
        happiness = min(100, happiness + 20)

    return {
        "reply_text": _DUMMY_REPLIES.get(current_state, _DUMMY_REPLIES["GREETING"]),
        "negotiation_state": next_state,
        "happiness_score": happiness,
        "vendor_mood": "friendly" if happiness >= 50 else "annoyed",
    }


# ── Use dummy brain for now (swap to Dev A's real function later) ─────
generate_vendor_response = _dummy_generate_vendor_response
logger.info("Using DUMMY brain for testing — Dev A's brain not connected")


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
    """Payload from Unity frontend — matches C# MinimalRequest."""
    audioData: str = Field(..., description="Base64-encoded mono WAV (16 kHz)")
    inputLanguage: str = Field("en", description="Source language code")
    targetLanguage: str = Field("en", description="Target language code")
    object_grabbed: str = Field("", description="Item the user is holding")
    happiness_score: int = Field(50, description="Vendor happiness 0-100")
    negotiation_state: str = Field("GREETING", description="Current negotiation stage")


@app.post("/api/test")
async def test_pipeline(req: TestRequest):
    """
    Main pipeline endpoint — matches Unity's MinimalRequest → BackendFullResponse.

    Receives:
    {
        "audioData": "<base64 mono WAV 16kHz>",
        "inputLanguage": "en",
        "targetLanguage": "en",
        "object_grabbed": "silk_scarf",
        "happiness_score": 50,
        "negotiation_state": "GREETING"
    }

    Returns:
    {
        "reply": "<transcribed user speech>",
        "audioReply": "<base64 TTS audio of vendor response>",
        "negotiation_state": "HAGGLING"
    }
    """
    request_start = time.perf_counter()
    session_id = "test-session"
    language_code = normalize_language_code(req.inputLanguage)
    target_language = normalize_language_code(req.targetLanguage)

    # ── Log incoming request ──
    logger.info("─" * 60)
    logger.info("▶ REQUEST  /api/test")
    logger.info("  inputLanguage     : %s → %s", req.inputLanguage, language_code)
    logger.info("  targetLanguage    : %s → %s", req.targetLanguage, target_language)
    logger.info("  object_grabbed    : %s", req.object_grabbed)
    logger.info("  happiness_score   : %d", req.happiness_score)
    logger.info("  negotiation_state : %s", req.negotiation_state)
    logger.info("  audioData         : %d chars", len(req.audioData))

    # ── Step 2 — Decode base64 → raw bytes ──
    t0 = time.perf_counter()
    try:
        audio_bytes = base64_to_bytes(req.audioData)
    except ValueError as e:
        logger.error("Step 2 FAILED: Invalid audioData — %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid audioData: {e}")
    logger.info("  Step 2 decode : %d bytes (%.1fms)", len(audio_bytes), (time.perf_counter() - t0) * 1000)

    # ── Step 3 — STT ──
    t0 = time.perf_counter()
    try:
        transcribed_text = await transcribe_with_sarvam(audio_bytes, language_code)
    except SarvamServiceError as e:
        logger.error("Step 3 FAILED: STT error — %s", e)
        raise HTTPException(status_code=503, detail=f"STT failed: {e}")
    logger.info("  Step 3 STT    : \"%s\" (%.0fms)", transcribed_text[:80], (time.perf_counter() - t0) * 1000)

    # Silence detected — short-circuit
    if transcribed_text == "":
        elapsed = (time.perf_counter() - request_start) * 1000
        logger.info("  Silence detected — returning prompt (%.0fms total)", elapsed)
        return {
            "reply": "",
            "audioReply": "",
            "negotiation_state": req.negotiation_state,
        }

    # ── Step 4 — Store user turn in memory ──
    memory = get_memory(session_id)
    logger.info("  Step 4 memory : session=%s, total_turns=%d", session_id, len(memory.get_recent_turns(n=100)))
    memory.add_turn(
        role="user",
        text=transcribed_text,
        metadata={
            "object_grabbed": req.object_grabbed,
            "happiness_score": req.happiness_score,
            "negotiation_state": req.negotiation_state,
        },
    )

    # ── Steps 5 & 6 — Parallel: context history + RAG retrieval ──
    t0 = time.perf_counter()
    rag_context = ""
    try:
        context_block, rag_context = await asyncio.gather(
            asyncio.to_thread(memory.get_context_block),
            retrieve_context(transcribed_text, n_results=3),
        )
    except RAGServiceError as e:
        context_block = memory.get_context_block()
        rag_context = ""
        logger.warning("RAG unavailable (%s), continuing without context", e)

    logger.info("  Steps 5+6     : context=%d chars, rag=%d chars (%.0fms)",
                len(context_block), len(rag_context), (time.perf_counter() - t0) * 1000)
    if rag_context:
        logger.info("  rag_preview   : %s", rag_context[:200])

    # ── Step 7 — Brain (mock or Dev A's real function) ──
    # Build scene_context dict from Unity's flat fields
    scene_context = {
        "object_grabbed": req.object_grabbed,
        "happiness_score": req.happiness_score,
        "negotiation_state": req.negotiation_state,
        "targetLanguage": req.targetLanguage,
    }

    t0 = time.perf_counter()
    logger.info("  Step 7 brain  : calling generate_vendor_response...")
    logger.info("    input text    : %s", transcribed_text[:100])
    logger.info("    scene_context : %s", json.dumps(scene_context, ensure_ascii=False))
    try:
        brain_result = await generate_vendor_response(
            transcribed_text=transcribed_text,
            context_block=context_block,
            rag_context=rag_context,
            scene_context=scene_context,
            session_id=session_id,
        )
    except Exception as e:
        logger.error("Step 7 FAILED: Brain error — %s", e)
        raise HTTPException(status_code=500, detail=f"Brain error: {e}")
    logger.info("  Step 7 result : %s (%.0fms)", json.dumps(brain_result, ensure_ascii=False)[:300],
                (time.perf_counter() - t0) * 1000)

    # Extract updated state from Dev A's response
    updated_negotiation_state = brain_result.get("negotiation_state", req.negotiation_state)

    # ── Step 8 — Store vendor turn in memory ──
    memory.add_turn(
        role="vendor",
        text=brain_result["reply_text"],
        metadata={
            "vendor_mood": brain_result.get("vendor_mood", "neutral"),
            "happiness_score": brain_result.get("happiness_score", req.happiness_score),
            "negotiation_state": updated_negotiation_state,
        },
    )

    # ── Steps 9 & 10 — TTS → base64 ──
    t0 = time.perf_counter()
    audio_reply = ""
    try:
        tts_bytes = await speak_with_sarvam(brain_result["reply_text"], target_language)
        audio_reply = bytes_to_base64(tts_bytes)
        logger.info("  Steps 9+10    : TTS %d bytes → %d chars b64 (%.0fms)",
                    len(tts_bytes), len(audio_reply), (time.perf_counter() - t0) * 1000)
    except SarvamServiceError as e:
        logger.warning("Steps 9+10 FAILED: TTS unavailable — %s (sending text-only)", e)

    # ── Step 11 — Return BackendFullResponse to Unity ──
    response = {
        "reply": transcribed_text,
        "audioReply": audio_reply,
        "negotiation_state": updated_negotiation_state,
    }

    # ── Log outgoing response ──
    total_ms = (time.perf_counter() - request_start) * 1000
    logger.info("◀ RESPONSE /api/test (%.0fms total)", total_ms)
    logger.info("  transcribed        : %s", transcribed_text)
    logger.info("  vendor_reply       : %s", brain_result["reply_text"])
    logger.info("  audio_out          : %d chars", len(audio_reply))
    logger.info("  negotiation_state  : %s → %s", req.negotiation_state, updated_negotiation_state)
    logger.info("─" * 60)

    return response


# ── Main endpoint ─────────────────────────────────────────────────────
@app.post("/api/interact", response_model=InteractResponse)
async def interact(request: InteractRequest) -> InteractResponse:
    """
    Full pipeline: Audio in → STT → Memory → RAG → LLM (Dev A) → TTS → Audio out.
    Steps 1–11 as defined in execution_plan.md.
    """
    request_start = time.perf_counter()

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
    t0 = time.perf_counter()
    try:
        audio_bytes = base64_to_bytes(request.audio_base64)
    except ValueError as e:
        logger.error("Step 2 FAILED: Invalid audio data — %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {e}")
    logger.info("  Step 2 decode : %d bytes (%.1fms)", len(audio_bytes), (time.perf_counter() - t0) * 1000)

    # Step 3 — Speech-to-Text
    t0 = time.perf_counter()
    try:
        transcribed_text = await transcribe_with_sarvam(
            audio_bytes, request.language_code
        )
    except SarvamServiceError as e:
        logger.error("Step 3 FAILED: STT error — %s", e)
        raise HTTPException(status_code=503, detail=f"Voice recognition unavailable: {e}")
    logger.info("  Step 3 STT    : \"%s\" (%.0fms)", transcribed_text[:80], (time.perf_counter() - t0) * 1000)

    # Silence detected — short-circuit
    if transcribed_text == "":
        elapsed = (time.perf_counter() - request_start) * 1000
        logger.info("  Silence detected — short-circuit (%.0fms total)", elapsed)
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
    t0 = time.perf_counter()
    try:
        context_block, rag_context = await asyncio.gather(
            asyncio.to_thread(memory.get_context_block),
            retrieve_context(transcribed_text, n_results=3),
        )
    except RAGServiceError as e:
        # Graceful degradation — continue without RAG
        context_block = memory.get_context_block()
        rag_context = ""
        logger.warning("RAG unavailable (%s), continuing without context", e)

    logger.info("  Steps 5+6     : context=%d chars, rag=%d chars (%.0fms)",
                len(context_block), len(rag_context), (time.perf_counter() - t0) * 1000)
    if rag_context:
        logger.info("  rag_preview   : %s", rag_context[:200])

    # Step 7 — Call Dev A's brain (LLM + Neo4j validation)
    t0 = time.perf_counter()
    logger.info("  Step 7 brain  : calling generate_vendor_response...")
    try:
        result = await generate_vendor_response(
            transcribed_text=transcribed_text,
            context_block=context_block,
            rag_context=rag_context,
            scene_context=request.scene_context.model_dump(),
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error("Step 7 FAILED: Brain/LLM error — %s", e)
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    logger.info("  Step 7 result : %s (%.0fms)", json.dumps(result, ensure_ascii=False)[:300],
                (time.perf_counter() - t0) * 1000)

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
    t0 = time.perf_counter()
    agent_audio_base64 = ""
    try:
        tts_bytes = await speak_with_sarvam(
            result["reply_text"], request.language_code
        )
        # Step 10 — Encode audio
        agent_audio_base64 = bytes_to_base64(tts_bytes)
        logger.info("  Steps 9+10    : TTS %d bytes → %d chars b64 (%.0fms)",
                    len(tts_bytes), len(agent_audio_base64), (time.perf_counter() - t0) * 1000)
    except SarvamServiceError as e:
        # Subtitle mode — text-only response
        logger.warning("Steps 9+10 FAILED: TTS unavailable — %s (sending text-only)", e)

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
    total_ms = (time.perf_counter() - request_start) * 1000
    logger.info("═" * 60)
    logger.info("◀ RESPONSE /api/interact (%.0fms total)", total_ms)
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
