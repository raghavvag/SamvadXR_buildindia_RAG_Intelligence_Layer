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
import uuid
from contextlib import asynccontextmanager

import httpx

from dotenv import load_dotenv
load_dotenv()  # Must run BEFORE importing services that read env vars

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from models import InteractRequest, InteractResponse, NegotiationState
from services.middleware import base64_to_bytes, bytes_to_base64
from services.voice_ops import transcribe_with_sarvam, speak_with_sarvam, translate_with_sarvam, detect_language_with_sarvam, detect_language_robust, normalize_language_code, get_language_name
from services.rag_ops import initialize_knowledge_base, retrieve_context, add_conversation_to_rag, clear_conversation_from_rag
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
_active_session_id: str | None = None


def get_memory(session_id: str) -> ConversationMemory:
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory(session_id=session_id)
    return sessions[session_id]


def get_or_create_session(negotiation_state: str) -> tuple[str, ConversationMemory, bool]:
    """
    Resolve session based on negotiation_state from Unity.

    - "" (empty string) → fresh session (new UUID, empty memory)
    - "GREETING" with NO active session → fresh session
    - "GREETING" with active session → CONTINUE (don't reset!)
    - Anything else → continue the active session

    Returns: (session_id, memory, is_new_session)
    """
    global _active_session_id

    # Only truly fresh when state is empty string, or GREETING with no prior session
    is_fresh = (
        negotiation_state == ""
        or (negotiation_state == "GREETING" and _active_session_id is None)
    )

    if is_fresh:
        new_id = f"session-{uuid.uuid4().hex[:8]}"
        sessions[new_id] = ConversationMemory(session_id=new_id)
        _active_session_id = new_id
        logger.info("  \U0001f195 New session: %s (state=%s)", new_id, negotiation_state or "<empty>")
        return new_id, sessions[new_id], True

    if _active_session_id and _active_session_id in sessions:
        logger.info("  \U0001f504 Continuing session: %s (state=%s, turns=%d)",
                    _active_session_id, negotiation_state,
                    len(sessions[_active_session_id].get_recent_turns(n=100)))
        return _active_session_id, sessions[_active_session_id], False

    # No active session but mid-flow state — create one anyway
    new_id = f"session-{uuid.uuid4().hex[:8]}"
    sessions[new_id] = ConversationMemory(session_id=new_id)
    _active_session_id = new_id
    logger.warning("  \u26a0\ufe0f No active session for state=%s, creating: %s", negotiation_state, new_id)
    return new_id, sessions[new_id], True


def reset_active_session() -> str:
    """Force-reset the active session. Returns the new session ID."""
    global _active_session_id
    old_id = _active_session_id
    _active_session_id = None
    logger.info("  🔄 Active session reset (was=%s)", old_id or "None")
    return old_id or ""


# ── Dummy brain for testing (skip Dev A for now) ──────────────────────
_DUMMY_REPLIES = {
    "GREETING":    "Arre sahab! Aao aao, aaj sabzi-fruit sab taaza aaya hai!",
    "INQUIRY":     "Yeh bilkul taaza hai sahab, subah mandi se aaya hai. Chhukar dekhiye!",
    "BARGAINING":  "Itna kam? Sahab yeh toh cost price se bhi kam hai! Chaliye thoda aur badhaaiye.",
    "COUNTER":     "Achha chaliye aapke liye special rate — ₹50 per kilo, final offer!",
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
        "suggested_user_response": "Bhaiya, thoda aur kam karo na?",
    }


# ── Use dummy brain for now (swap to Dev A's real function later) ─────
# generate_vendor_response = _dummy_generate_vendor_response
# logger.info("Using DUMMY brain for testing — Dev A's brain not connected")

# ── Dev A's real brain endpoint ───────────────────────────────────────
DEV_A_URL = os.getenv(
    "DEV_A_URL",
    "https://7ec4-2409-40f4-30-ed85-1cc7-f2da-f7dc-a771.ngrok-free.app/api/dev/generate",
)
_DEV_A_TIMEOUT = float(os.getenv("DEV_A_TIMEOUT", "30"))  # seconds


async def generate_vendor_response(
    session_id: str,
    transcribed_text: str,
    context_block: str,
    rag_context: str,
    scene_context: dict,
) -> dict:
    """
    Call Dev A's /dev/generate endpoint.
    Falls back to dummy brain if the call fails.
    """
    # Prepend shop-type instruction so Dev A's LLM knows the setting
    shop_instruction = (
        "[IMPORTANT INSTRUCTION] You are a sabzi mandi (vegetable market) "
        "and fruit shop vendor. You sell ONLY fruits and vegetables — "
        "NO silk, NO brass, NO handicrafts, NO everyday goods. "
        "Never mention silk, brass, or any non-produce items.\n\n"
    )
    payload = {
        "session_id": session_id,
        "transcribed_text": transcribed_text,
        "context_block": shop_instruction + context_block,
        "rag_context": rag_context,
        "scene_context": scene_context,
    }
    logger.info("  → Calling Dev A at %s", DEV_A_URL)
    logger.info("    payload.session_id       : %s", payload["session_id"])
    logger.info("    payload.transcribed_text  : %s", payload["transcribed_text"][:150])
    logger.info("    payload.context_block     : %d chars", len(payload["context_block"]))
    logger.info("    payload.rag_context       : %d chars", len(payload["rag_context"]))
    logger.info("    payload.scene_context     : %s", json.dumps(payload["scene_context"], ensure_ascii=False))

    try:
        async with httpx.AsyncClient(timeout=_DEV_A_TIMEOUT) as client:
            resp = await client.post(DEV_A_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            logger.info("  ✅ Dev A responded (status=%s)", resp.status_code)
            logger.info("    Dev A raw keys: %s", list(data.keys()))
            logger.info("    Dev A raw response: %s", json.dumps(data, ensure_ascii=False)[:500])

            # Unwrap if Dev A nests inside "data" or "response"
            if "data" in data and isinstance(data["data"], dict):
                data = data["data"]
                logger.info("    ↳ unwrapped from 'data' key")
            elif "response" in data and isinstance(data["response"], dict):
                data = data["response"]
                logger.info("    ↳ unwrapped from 'response' key")

            # Normalise keys — Dev A may return slightly different names
            suggested = (
                data.get("suggested_user_response")
                or data.get("suggested_response")
                or data.get("suggestedResponse")
                or ""
            )
            return {
                "reply_text": data.get("reply_text", data.get("reply", "")),
                "negotiation_state": data.get("negotiation_state", scene_context.get("negotiation_state", "GREETING")),
                "happiness_score": data.get("happiness_score", scene_context.get("happiness_score", 50)),
                "vendor_mood": data.get("vendor_mood", "neutral"),
                "suggested_user_response": suggested,
            }

    except httpx.TimeoutException:
        logger.error("  ⏱️ Dev A timed out after %.1fs — falling back to dummy brain", _DEV_A_TIMEOUT)
    except httpx.HTTPStatusError as exc:
        logger.error("  ❌ Dev A returned HTTP %s — falling back to dummy brain", exc.response.status_code)
    except Exception as exc:
        logger.error("  ❌ Dev A call failed (%s: %s) — falling back to dummy brain", type(exc).__name__, exc)

    # ── Fallback to dummy brain ───────────────────────────────────────
    logger.warning("  ⚠️ Using DUMMY brain fallback")
    return await _dummy_generate_vendor_response(
        transcribed_text=transcribed_text,
        context_block=context_block,
        rag_context=rag_context,
        scene_context=scene_context,
        session_id=session_id,
    )


logger.info("Dev A brain endpoint configured: %s (timeout=%.1fs)", DEV_A_URL, _DEV_A_TIMEOUT)

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


@app.post("/api/reset")
async def reset_session():
    """Explicitly reset the active session. Call this when starting a new VR scene."""
    old_id = reset_active_session()
    # Clear conversation chunks from RAG for the old session
    if old_id:
        await clear_conversation_from_rag(old_id)
    return {
        "status": "ok",
        "message": "Session reset — next request will create a fresh session",
        "old_session_id": old_id,
    }


@app.post("/api/reload-rag")
async def reload_rag():
    """
    Force-reload the RAG knowledge base from disk.
    Wipes ALL ChromaDB data (seed + conversation chunks) and re-embeds
    the current .txt files. Use after updating data/ files.
    """
    logger.info("🔁 Force-reloading RAG knowledge base...")
    initialize_knowledge_base()
    logger.info("✅ RAG knowledge base reloaded from disk.")
    return {
        "status": "ok",
        "message": "RAG knowledge base reloaded from current data/ files",
    }


@app.get("/api/debug/rag")
async def debug_rag():
    """
    Dump all documents currently in ChromaDB for inspection.
    Use this to verify no stale silk/brass data exists.
    """
    from services.rag_ops import _collection
    if _collection is None:
        return {"status": "error", "message": "ChromaDB collection not initialized"}

    all_data = _collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents", [])
    ids = all_data.get("ids", [])
    metas = all_data.get("metadatas", [])

    # Check for forbidden keywords
    forbidden = ["silk", "brass", "pashmina", "handicraft"]
    contaminated = []
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        found = [kw for kw in forbidden if kw in doc_lower]
        if found:
            contaminated.append({
                "id": ids[i],
                "keywords_found": found,
                "preview": doc[:200],
                "source": metas[i].get("source", "unknown") if i < len(metas) else "unknown",
            })

    return {
        "status": "ok",
        "total_documents": len(docs),
        "contaminated_count": len(contaminated),
        "contaminated": contaminated,
        "all_sources": list(set(
            m.get("source", "unknown") for m in metas
        )) if metas else [],
    }


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
        "object_grabbed": "tomato",
        "happiness_score": 50,
        "negotiation_state": "GREETING"
    }

    Returns:
    {
        "reply": "<vendor's text response>",
        "audioReply": "<base64 TTS audio of vendor response>",
        "negotiation_state": "HAGGLING",
        "happiness_score": 55
    }
    """
    request_start = time.perf_counter()
    session_id, memory, is_new = get_or_create_session(req.negotiation_state)
    target_language = normalize_language_code(req.targetLanguage)

    # ── Log incoming request ──
    logger.info("─" * 60)
    logger.info("▶ REQUEST  /api/test")
    logger.info("  session_id        : %s %s", session_id, "(NEW)" if is_new else "(continuing)")
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

    # ── Step 3 — STT (use target_language as hint — Sarvam works best with it) ──
    t0 = time.perf_counter()
    try:
        transcribed_text = await transcribe_with_sarvam(audio_bytes, target_language)
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
            "happiness_score": req.happiness_score,
            "suggested_user_response": "",
        }

    # ── Step 3.5A — Detect actual language from transcribed text ──
    # Uses script detection (Latin vs Devanagari etc.) + Sarvam API fallback
    t0 = time.perf_counter()
    detected_language = await detect_language_robust(transcribed_text, target_language)
    logger.info("  Step 3.5A detect : %s (target=%s) (%.0fms)",
                detected_language, target_language, (time.perf_counter() - t0) * 1000)

    # ── Step 3.5B — LANGUAGE GATE ──
    # If the user is NOT speaking the target language, reject early.
    # Send rejection in the TARGET language (user chose it, so they understand it).
    if detected_language != target_language:
        logger.info("  ╔══ LANGUAGE MISMATCH ══════════════════════════════════")
        logger.info("  ║ detected_language : %s (%s)", detected_language, get_language_name(detected_language))
        logger.info("  ║ target_language   : %s (%s)", target_language, get_language_name(target_language))
        logger.info("  ║ action            : REJECT — send message in target language")
        logger.info("  ╚══════════════════════════════════════════════════════")

        # Build rejection message in English, then translate to TARGET language
        target_name = get_language_name(target_language)
        rejection_english = f"You need to speak in {target_name}. Please try again in {target_name}."
        logger.info("  Step 3.5B rejection (en): %s", rejection_english)

        # Translate the rejection into the TARGET language
        t0 = time.perf_counter()
        rejection_text = await translate_with_sarvam(rejection_english, target_language)
        logger.info("  Step 3.5B rejection (%s): %s (%.0fms)",
                    target_language, rejection_text, (time.perf_counter() - t0) * 1000)

        # TTS the rejection in TARGET language
        t0 = time.perf_counter()
        rejection_audio = ""
        try:
            tts_bytes = await speak_with_sarvam(rejection_text, target_language)
            rejection_audio = bytes_to_base64(tts_bytes)
            logger.info("  Step 3.5B TTS (%s): %d bytes → %d chars b64 (%.0fms)",
                        target_language, len(tts_bytes), len(rejection_audio), (time.perf_counter() - t0) * 1000)
        except SarvamServiceError as e:
            logger.warning("  Step 3.5B TTS FAILED: %s (text-only fallback)", e)

        total_ms = (time.perf_counter() - request_start) * 1000
        logger.info("  ╔══ RESPONSE SUMMARY (MISMATCH) ══════════════════════")
        logger.info("  ║ reply             : %s", rejection_text)
        logger.info("  ║ audio_out         : %d chars", len(rejection_audio))
        logger.info("  ║ negotiation_state : %s (unchanged)", req.negotiation_state)
        logger.info("  ║ happiness_score   : %d (unchanged)", req.happiness_score)
        logger.info("  ║ total_time        : %.0fms", total_ms)
        logger.info("  ╚══════════════════════════════════════════════════════")
        logger.info("◀ RESPONSE /api/test — LANGUAGE MISMATCH (%.0fms)", total_ms)
        logger.info("─" * 60)

        return {
            "reply": rejection_text,
            "audioReply": rejection_audio,
            "negotiation_state": req.negotiation_state,   # state unchanged
            "happiness_score": req.happiness_score,       # score unchanged
            "suggested_user_response": "",
        }

    # ── Language matches — proceed with full pipeline ──
    logger.info("  ✅ Language match: detected=%s == target=%s", detected_language, target_language)

    # ── Step 3.5C — Translate user's text to English for Dev A ──
    t0 = time.perf_counter()
    transcribed_text_original = transcribed_text
    transcribed_text_english = await translate_with_sarvam(transcribed_text, "en-IN")
    if transcribed_text_english != transcribed_text:
        logger.info("  Step 3.5C translate→en: \"%s\" → \"%s\" (%.0fms)",
                    transcribed_text[:60], transcribed_text_english[:60],
                    (time.perf_counter() - t0) * 1000)
    else:
        transcribed_text_english = transcribed_text
        logger.info("  Step 3.5C translate→en: already English (%.0fms)", (time.perf_counter() - t0) * 1000)

    # ── Step 4 — Store user turn in memory (original language) ──
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
    # Use English text for RAG (better semantic matching with English seed data)
    t0 = time.perf_counter()
    rag_context = ""
    try:
        context_block, rag_context = await asyncio.gather(
            asyncio.to_thread(memory.get_context_block),
            retrieve_context(transcribed_text_english, n_results=5),
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

    # ── Step 7 — Brain (Dev A) ──
    # Send target_language from frontend (NOT detected) as both input_language and target_language
    scene_context = {
        "object_grabbed": req.object_grabbed,
        "happiness_score": req.happiness_score,
        "negotiation_state": req.negotiation_state,
        "input_language": target_language,
        "target_language": target_language,
    }

    dev_a_payload = {
        "session_id": session_id,
        "transcribed_text": transcribed_text_english,
        "context_block": context_block[:200] + "..." if len(context_block) > 200 else context_block,
        "rag_context": rag_context[:200] + "..." if len(rag_context) > 200 else rag_context,
        "scene_context": scene_context,
    }

    t0 = time.perf_counter()
    logger.info("  ╔══ STEP 7: DEV A REQUEST ═════════════════════════════")
    logger.info("  ║ endpoint          : %s", DEV_A_URL)
    logger.info("  ║ session_id        : %s", session_id)
    logger.info("  ║ transcribed (en)  : %s", transcribed_text_english[:120])
    logger.info("  ║ context_block     : %d chars", len(context_block))
    logger.info("  ║ rag_context       : %d chars", len(rag_context))
    logger.info("  ║ scene_context     : %s", json.dumps(scene_context, ensure_ascii=False))
    logger.info("  ║ FULL PAYLOAD (preview): %s", json.dumps(dev_a_payload, ensure_ascii=False)[:500])
    logger.info("  ╚══════════════════════════════════════════════════════")
    try:
        brain_result = await generate_vendor_response(
            transcribed_text=transcribed_text_english,
            context_block=context_block,
            rag_context=rag_context,
            scene_context=scene_context,
            session_id=session_id,
        )
    except Exception as e:
        logger.error("Step 7 FAILED: Brain error — %s", e)
        raise HTTPException(status_code=500, detail=f"Brain error: {e}")
    logger.info("  ╔══ STEP 7: DEV A RESPONSE ════════════════════════════")
    logger.info("  ║ reply_text        : %s", brain_result.get('reply_text', '')[:200])
    logger.info("  ║ negotiation_state : %s", brain_result.get('negotiation_state', 'N/A'))
    logger.info("  ║ happiness_score   : %s", brain_result.get('happiness_score', 'N/A'))
    logger.info("  ║ vendor_mood       : %s", brain_result.get('vendor_mood', 'N/A'))
    logger.info("  ║ suggested_response: %s", brain_result.get('suggested_user_response', '')[:100])
    logger.info("  ║ latency           : %.0fms", (time.perf_counter() - t0) * 1000)
    logger.info("  ╚══════════════════════════════════════════════════════")

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

    # ── Step 8.1 — Update RAG with this conversation turn ──
    turn_count = len(memory.get_recent_turns(n=100))
    await add_conversation_to_rag(
        session_id=session_id,
        turn_index=turn_count // 2,  # pair count
        user_text=transcribed_text_english,
        vendor_text=brain_result["reply_text"],
        object_grabbed=req.object_grabbed,
    )

    # ── Step 8.5 — Translate Dev A's English reply → target_language ──
    t0 = time.perf_counter()
    vendor_text_original = brain_result["reply_text"]
    vendor_text = await translate_with_sarvam(vendor_text_original, target_language)
    translate_ms = (time.perf_counter() - t0) * 1000
    if vendor_text != vendor_text_original:
        logger.info("  Step 8.5 translate: en → %s (%.0fms)", target_language, translate_ms)
        logger.info("    original (en) : %s", vendor_text_original[:120])
        logger.info("    translated    : %s", vendor_text[:120])
    else:
        logger.info("  Step 8.5 translate: no change needed (%.0fms)", translate_ms)

    # ── Steps 9 & 10 — TTS in target_language → base64 ──
    t0 = time.perf_counter()
    audio_reply = ""
    try:
        tts_bytes = await speak_with_sarvam(vendor_text, target_language)
        audio_reply = bytes_to_base64(tts_bytes)
        logger.info("  Steps 9+10 TTS (%s): %d bytes → %d chars b64 (%.0fms)",
                    target_language, len(tts_bytes), len(audio_reply), (time.perf_counter() - t0) * 1000)
    except SarvamServiceError as e:
        logger.warning("  Steps 9+10 TTS FAILED: %s (text-only fallback)", e)

    # ── Step 11 — Return BackendFullResponse to Unity ──
    updated_happiness = brain_result.get("happiness_score", req.happiness_score)
    suggested_response = brain_result.get("suggested_user_response", "")
    response = {
        "reply": vendor_text,
        "audioReply": audio_reply,
        "negotiation_state": updated_negotiation_state,
        "happiness_score": updated_happiness,
        "suggested_user_response": suggested_response,
    }

    # ── End-to-end summary ──
    total_ms = (time.perf_counter() - request_start) * 1000
    logger.info("  ╔══ RESPONSE SUMMARY (/api/test) ═════════════════════")
    logger.info("  ║ total_time        : %.0fms", total_ms)
    logger.info("  ║ target_language   : %s (%s)", target_language, get_language_name(target_language))
    logger.info("  ║ detected_language : %s (%s)", detected_language, get_language_name(detected_language))
    logger.info("  ║ transcribed       : %s", transcribed_text_original[:120])
    logger.info("  ║ transcribed (en)  : %s", transcribed_text_english[:120])
    logger.info("  ║ vendor_reply (en) : %s", brain_result["reply_text"][:120])
    logger.info("  ║ vendor_reply (tgt): %s", vendor_text[:120])
    logger.info("  ║ audio_out         : %d chars", len(audio_reply))
    logger.info("  ║ negotiation_state : %s → %s", req.negotiation_state, updated_negotiation_state)
    logger.info("  ║ happiness_score   : %d → %d", req.happiness_score, updated_happiness)
    logger.info("  ║ suggested_response: %s", suggested_response[:100] if suggested_response else "(none)")
    logger.info("  ║ memory_turns      : %d", len(memory.get_recent_turns(n=100)))
    logger.info("  ║ active_sessions   : %d", len(sessions))
    logger.info("  ╚══════════════════════════════════════════════════════")
    logger.info("◀ RESPONSE /api/test (%.0fms)", total_ms)
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
    target_language = normalize_language_code(request.language_code)

    # ── Log incoming request ──
    logger.info("═" * 60)
    logger.info("▶ REQUEST  /api/interact")
    logger.info("  session_id      : %s", request.session_id)
    logger.info("  target_language : %s → %s", request.language_code, target_language)
    logger.info("  audio_base64    : %d chars", len(request.audio_base64))
    logger.info("  scene_context   : %s", json.dumps(request.scene_context.model_dump(), ensure_ascii=False))
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

    # Step 3 — Speech-to-Text (use target_language as hint)
    t0 = time.perf_counter()
    try:
        transcribed_text = await transcribe_with_sarvam(audio_bytes, target_language)
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

    # Step 3.5A — Detect actual language from transcribed text
    # Uses script detection (Latin vs Devanagari etc.) + Sarvam API fallback
    t0 = time.perf_counter()
    detected_language = await detect_language_robust(transcribed_text, target_language)
    logger.info("  Step 3.5A detect : %s (target=%s) (%.0fms)",
                detected_language, target_language, (time.perf_counter() - t0) * 1000)

    # ── Step 3.5B — LANGUAGE GATE ──
    # Send rejection in TARGET language (user chose it, so they understand it).
    if detected_language != target_language:
        logger.info("  ╔══ LANGUAGE MISMATCH ══════════════════════════════════")
        logger.info("  ║ detected_language : %s (%s)", detected_language, get_language_name(detected_language))
        logger.info("  ║ target_language   : %s (%s)", target_language, get_language_name(target_language))
        logger.info("  ║ action            : REJECT — send message in target language")
        logger.info("  ╚══════════════════════════════════════════════════════")

        target_name = get_language_name(target_language)
        rejection_english = f"You need to speak in {target_name}. Please try again in {target_name}."
        logger.info("  Step 3.5B rejection (en): %s", rejection_english)

        t0 = time.perf_counter()
        rejection_text = await translate_with_sarvam(rejection_english, target_language)
        logger.info("  Step 3.5B rejection (%s): %s (%.0fms)",
                    target_language, rejection_text, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        rejection_audio = ""
        try:
            tts_bytes = await speak_with_sarvam(rejection_text, target_language)
            rejection_audio = bytes_to_base64(tts_bytes)
            logger.info("  Step 3.5B TTS (%s): %d bytes → %d chars b64 (%.0fms)",
                        target_language, len(tts_bytes), len(rejection_audio), (time.perf_counter() - t0) * 1000)
        except SarvamServiceError as e:
            logger.warning("  Step 3.5B TTS FAILED: %s (text-only fallback)", e)

        total_ms = (time.perf_counter() - request_start) * 1000
        logger.info("  ╔══ RESPONSE SUMMARY (MISMATCH) ══════════════════════")
        logger.info("  ║ reply             : %s", rejection_text)
        logger.info("  ║ audio_out         : %d chars", len(rejection_audio))
        logger.info("  ║ total_time        : %.0fms", total_ms)
        logger.info("  ╚══════════════════════════════════════════════════════")
        logger.info("◀ RESPONSE /api/interact — LANGUAGE MISMATCH (%.0fms)", total_ms)
        logger.info("═" * 60)

        return InteractResponse(
            session_id=request.session_id,
            transcribed_text=transcribed_text,
            agent_reply_text=rejection_text,
            agent_audio_base64=rejection_audio,
            vendor_mood="confused",
        )

    # ── Language matches — proceed with full pipeline ──
    logger.info("  ✅ Language match: detected=%s == target=%s", detected_language, target_language)

    # Step 3.5C — Translate to English for Dev A + RAG
    t0 = time.perf_counter()
    transcribed_text_original = transcribed_text
    transcribed_text_english = await translate_with_sarvam(transcribed_text, "en-IN")
    if transcribed_text_english != transcribed_text:
        logger.info("  Step 3.5C translate→en: \"%s\" (%.0fms)",
                    transcribed_text_english[:80], (time.perf_counter() - t0) * 1000)
    else:
        transcribed_text_english = transcribed_text
        logger.info("  Step 3.5C translate→en: already English (%.0fms)", (time.perf_counter() - t0) * 1000)

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
            retrieve_context(transcribed_text_english, n_results=5),
        )
    except RAGServiceError as e:
        context_block = memory.get_context_block()
        rag_context = ""
        logger.warning("RAG unavailable (%s), continuing without context", e)

    logger.info("  Steps 5+6     : context=%d chars, rag=%d chars (%.0fms)",
                len(context_block), len(rag_context), (time.perf_counter() - t0) * 1000)
    if rag_context:
        logger.info("  rag_preview   : %s", rag_context[:200])

    # Step 7 — Call Dev A's brain
    # Send target_language from frontend as both input_language and target_language
    enriched_scene_context = request.scene_context.model_dump()
    enriched_scene_context["input_language"] = target_language
    enriched_scene_context["target_language"] = target_language

    dev_a_payload_preview = {
        "session_id": request.session_id,
        "transcribed_text": transcribed_text_english,
        "context_block": context_block[:200] + "..." if len(context_block) > 200 else context_block,
        "rag_context": rag_context[:200] + "..." if len(rag_context) > 200 else rag_context,
        "scene_context": enriched_scene_context,
    }

    t0 = time.perf_counter()
    logger.info("  ╔══ STEP 7: DEV A REQUEST ═════════════════════════════")
    logger.info("  ║ endpoint          : %s", DEV_A_URL)
    logger.info("  ║ session_id        : %s", request.session_id)
    logger.info("  ║ transcribed (en)  : %s", transcribed_text_english[:120])
    logger.info("  ║ context_block     : %d chars", len(context_block))
    logger.info("  ║ rag_context       : %d chars", len(rag_context))
    logger.info("  ║ scene_context     : %s", json.dumps(enriched_scene_context, ensure_ascii=False))
    logger.info("  ║ FULL PAYLOAD (preview): %s", json.dumps(dev_a_payload_preview, ensure_ascii=False)[:500])
    logger.info("  ╚══════════════════════════════════════════════════════")
    try:
        result = await generate_vendor_response(
            transcribed_text=transcribed_text_english,
            context_block=context_block,
            rag_context=rag_context,
            scene_context=enriched_scene_context,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error("Step 7 FAILED: Brain/LLM error — %s", e)
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    logger.info("  ╔══ STEP 7: DEV A RESPONSE ════════════════════════════")
    logger.info("  ║ reply_text        : %s", result.get('reply_text', '')[:200])
    logger.info("  ║ negotiation_state : %s", result.get('negotiation_state', 'N/A'))
    logger.info("  ║ happiness_score   : %s", result.get('happiness_score', 'N/A'))
    logger.info("  ║ vendor_mood       : %s", result.get('vendor_mood', 'N/A'))
    logger.info("  ║ suggested_response: %s", result.get('suggested_user_response', '')[:100])
    logger.info("  ║ latency           : %.0fms", (time.perf_counter() - t0) * 1000)
    logger.info("  ╚══════════════════════════════════════════════════════")

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

    # Step 8.1 — Update RAG with this conversation turn
    turn_count = len(memory.get_recent_turns(n=100))
    await add_conversation_to_rag(
        session_id=request.session_id,
        turn_index=turn_count // 2,
        user_text=transcribed_text_english,
        vendor_text=result["reply_text"],
        object_grabbed=(
            request.scene_context.items_in_hand[0]
            if request.scene_context.items_in_hand
            else ""
        ),
    )

    # Step 8.5 — Translate Dev A's English reply → target_language
    t0 = time.perf_counter()
    reply_text_original = result["reply_text"]
    reply_text = await translate_with_sarvam(reply_text_original, target_language)
    translate_ms = (time.perf_counter() - t0) * 1000
    if reply_text != reply_text_original:
        logger.info("  Step 8.5 translate: en → %s (%.0fms)", target_language, translate_ms)
        logger.info("    original (en) : %s", reply_text_original[:120])
        logger.info("    translated    : %s", reply_text[:120])
    else:
        logger.info("  Step 8.5 translate: no change needed (%.0fms)", translate_ms)

    # Steps 9+10 — TTS in target_language → base64
    t0 = time.perf_counter()
    agent_audio_base64 = ""
    try:
        tts_bytes = await speak_with_sarvam(reply_text, target_language)
        agent_audio_base64 = bytes_to_base64(tts_bytes)
        logger.info("  Steps 9+10 TTS (%s): %d bytes → %d chars b64 (%.0fms)",
                    target_language, len(tts_bytes), len(agent_audio_base64), (time.perf_counter() - t0) * 1000)
    except SarvamServiceError as e:
        logger.warning("  Steps 9+10 TTS FAILED: %s (text-only fallback)", e)

    # Step 11 — Return response to Unity
    resp = InteractResponse(
        session_id=request.session_id,
        transcribed_text=transcribed_text,
        agent_reply_text=reply_text,
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

    # ── End-to-end summary ──
    total_ms = (time.perf_counter() - request_start) * 1000
    logger.info("  ╔══ RESPONSE SUMMARY (/api/interact) ═════════════════")
    logger.info("  ║ total_time        : %.0fms", total_ms)
    logger.info("  ║ target_language   : %s (%s)", target_language, get_language_name(target_language))
    logger.info("  ║ detected_language : %s (%s)", detected_language, get_language_name(detected_language))
    logger.info("  ║ transcribed       : %s", transcribed_text_original[:120])
    logger.info("  ║ transcribed (en)  : %s", transcribed_text_english[:120])
    logger.info("  ║ reply_text (en)   : %s", result["reply_text"][:120])
    logger.info("  ║ reply_text (tgt)  : %s", reply_text[:120])
    logger.info("  ║ audio_out         : %d chars", len(agent_audio_base64))
    logger.info("  ║ vendor_mood       : %s", result.get("vendor_mood", "neutral"))
    logger.info("  ║ negotiation       : stage=%s, price=%s, happiness=%s",
                result.get("new_stage"), result.get("price_offered"), result.get("vendor_happiness"))
    logger.info("  ╚══════════════════════════════════════════════════════")
    logger.info("◀ RESPONSE /api/interact (%.0fms)", total_ms)
    logger.info("═" * 60)

    return resp


# ── Run with uvicorn ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
