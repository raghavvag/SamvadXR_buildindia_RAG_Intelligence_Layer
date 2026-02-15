# Samvad XR — Data Flow Workflow
## How Data Travels Between Developer A & Developer B

---

## Overview

**Initial input from Unity:** Audio (Base64) + JSON (scene scores)
**Final output to Unity:** Audio (Base64) + JSON (vendor reply + state)

```
Dev B = "Senses, Memory & Orchestration" — API Endpoint, STT, TTS, RAG,
         Conversation Memory, Full Pipeline Control
Dev A = "Brain & Rules" — LLM Agent, Neo4j State Engine
         (called by Dev B as an internal service)
```

**Key change from v2.0:** Dev B owns the FastAPI endpoint. Unity talks directly to Dev B.
Dev A provides a function (`generate_vendor_response`) that Dev B calls mid-pipeline.

---

## The Raw Input (What Arrives from Unity)

Unity sends a single POST request to **Dev B's endpoint:**

```json
{
  "session_id": "vr-session-abc123",
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "language_code": "hi-IN",
  "scene_context": {
    "items_in_hand": ["brass_keychain"],
    "looking_at": "silk_scarf",
    "distance_to_vendor": 1.2,
    "vendor_npc_id": "vendor_01",
    "vendor_happiness": 55,
    "vendor_patience": 70,
    "negotiation_stage": "BROWSING",
    "current_price": 0,
    "user_offer": 0
  }
}
```

---

## Step-by-Step Data Flow

---

### STEP 1 — Receive & Parse Request
| | |
|---|---|
| **Who** | **Dev B** |
| **What** | Receives POST `/api/interact`, validates Pydantic model, extracts fields |
| **Input** | Raw HTTP request from Unity |
| **Output** | Parsed `session_id`, `audio_base64`, `language_code`, `scene_context` |
| **Error** | `422 Unprocessable Entity` if request body fails Pydantic validation |
| **Continues to** | Step 2 (Dev B continues) |

---

### STEP 2 — Decode Audio
| | |
|---|---|
| **Who** | **Dev B** |
| **Function** | `base64_to_bytes(audio_base64)` |
| **Input** | `"UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."` (base64 string, may have `data:audio/wav;base64,` prefix) |
| **Output** | `b'\x52\x49\x46\x46...'` (raw WAV bytes) |
| **Error** | `ValueError` if invalid base64 → Dev B returns 400 to Unity |
| **Continues to** | Step 3 (Dev B continues) |

---

### STEP 3 — Speech-to-Text (Sarvam STT)
| | |
|---|---|
| **Who** | **Dev B** |
| **Function** | `await transcribe_with_sarvam(audio_bytes, language_code)` |
| **Input** | Raw WAV bytes + `"hi-IN"` |
| **Output** | `"भाई ये silk scarf कितने का है?"` (native script string) |
| **On silence** | Returns `""` (empty string) → Dev B skips Steps 4-6, vendor says "Kuch bola?" |
| **On API down** | Raises `SarvamServiceError` → Dev B returns 503 to Unity |
| **On no API key** | Returns `"[MOCK] User said something in hi-IN"` (mock mode) |
| **Continues to** | Step 4 (Dev B continues) |

---

### STEP 4 — Store User Turn in Memory
| | |
|---|---|
| **Who** | **Dev B** |
| **Function** | `memory.add_turn(role, text, metadata)` |
| **Input** | |
```python
memory.add_turn(
    role="user",
    text="भाई ये silk scarf कितने का है?",     # ← from Step 3
    metadata={                                     # ← from Unity's scene_context
        "held_item": "brass_keychain",
        "looked_at_item": "silk_scarf",
        "vendor_happiness": 55,
        "vendor_patience": 70,
        "stage": "BROWSING"
    }
)
```
| **Output** | None (stored internally in the ConversationMemory instance) |
| **What happens inside** | Turn appended to `self._turns[]`. If window exceeded, older turns compressed into rolling summary. |

---

### STEP 5 — Get Conversation History
| | |
|---|---|
| **Who** | **Dev B** |
| **Function** | `memory.get_context_block()` |
| **Input** | None (reads from internal state) |
| **Output** | Text-only conversation history string: |
```
[Summary of earlier conversation]
The user greeted the vendor and asked about silk scarves.

[Recent Dialogue]
[Turn 1] User: Namaste bhaiya!
[Turn 1] Vendor: Aao aao! Kya chahiye?
[Turn 2] User: भाई ये silk scarf कितने का है?
```
| **Used by** | Dev B passes this to Dev A's function in Step 7 |

---

### STEP 6 — Retrieve Cultural Knowledge (RAG)
| | |
|---|---|
| **Who** | **Dev B** |
| **Function** | `await retrieve_context(query, n_results=3)` |
| **Input** | `"भाई ये silk scarf कितने का है?"` (the transcribed text from Step 3) |
| **Output** | Single string of relevant knowledge chunks: |
```
Silk Scarf: Wholesale ₹150, Fair Retail ₹300-400, Tourist Price ₹800-1200.
In Indian street markets, the vendor's first price is always 2-4 times the actual value. A good buyer starts at 25-30% of the quoted price.
```
| **On no results** | Returns `""` (empty string) |
| **On ChromaDB error** | Raises `RAGServiceError` → Dev B sets rag_context="" and continues (graceful) |
| **Used by** | Dev B passes this to Dev A's function in Step 7 |

> **Note:** Steps 5 and 6 are independent — Dev B runs them in parallel via `asyncio.gather()`.

---

### STEP 7 — LLM Agent Decides Response
| | |
|---|---|
| **Who** | **Dev A** (called by Dev B) |
| **Function** | Dev B calls Dev A's function: |
```python
result = await generate_vendor_response(
    transcribed_text="भाई ये silk scarf कितने का है?",   # ← from Step 3
    context_block=context_block,                           # ← from Step 5
    rag_context=rag_context,                               # ← from Step 6
    scene_context=request.scene_context,                   # ← from Unity
    session_id=request.session_id
)
```
| **What Dev A does internally** | Composes LLM prompt + calls LLM + queries Neo4j state |
| **Combined prompt looks like** | |
```
SYSTEM: You are Ramesh, a 55-year-old vendor in Jaipur bazaar...
CONVERSATION HISTORY: [from Dev B - Step 5]
CULTURAL CONTEXT: [from Dev B - Step 6]
GAME STATE: mood=55, stage=BROWSING, price_floor=300 [from Neo4j]
USER SAYS: भाई ये silk scarf कितने का है? [from Dev B - Step 3]
```
| **Returns to Dev B** | |
```python
{
    "reply_text": "अरे भाई, ये pure Banarasi silk है! ₹800 लगेगा",
    "new_mood": 60,
    "new_stage": "HAGGLING",
    "price_offered": 800,
    "vendor_happiness": 60,
    "vendor_patience": 68
}
```
| **On LLM error** | Raises exception → Dev B returns 500 to Unity |

---

### STEP 7½ — State Validation (Inside Dev A's Function)
| | |
|---|---|
| **Who** | **Dev A** (internal to Step 7 — Dev B doesn't see this separately) |
| **What** | Validates LLM output against Neo4j state rules before returning to Dev B |
| **Rules** | Mood clamped ±15 per turn, stage transitions must be legal, price within bounds |
| **Output** | The validated dict returned in Step 7 already includes clamped/validated values |

---

### STEP 8 — Store Vendor Turn in Memory
| | |
|---|---|
| **Who** | **Dev B** |
| **Function** | `memory.add_turn(role, text, metadata)` |
| **Input** | |
```python
memory.add_turn(
    role="vendor",
    text="अरे भाई, ये pure Banarasi silk है! ₹800 लगेगा",  # ← from Step 7 result
    metadata={                                                   # ← from Step 7 result
        "vendor_happiness": 60,
        "vendor_patience": 68,
        "stage": "HAGGLING",
        "price": 800
    }
)
```
| **Output** | None (stored internally) |

---

### STEP 9 — Text-to-Speech (Sarvam TTS)
| | |
|---|---|
| **Who** | **Dev B** |
| **Function** | `await speak_with_sarvam(text, language_code)` |
| **Input** | `"अरे भाई, ये pure Banarasi silk है! ₹800 लगेगा"` + `"hi-IN"` |
| **Output** | Raw WAV audio bytes (16-bit PCM, 22kHz, mono) |
| **On API down** | Raises `SarvamServiceError` → Dev B sends text-only response (subtitle mode, `audio_base64=""`) |
| **On no API key** | Returns 1-second silent WAV (valid file, just silence) |
| **Continues to** | Step 10 (Dev B continues) |

---

### STEP 10 — Encode Audio
| | |
|---|---|
| **Who** | **Dev B** |
| **Function** | `bytes_to_base64(audio_bytes)` |
| **Input** | Raw WAV bytes from Step 9 |
| **Output** | `"UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."` (base64 string) |
| **Continues to** | Step 11 (Dev B continues) |

---

### STEP 11 — Return Response to Unity
| | |
|---|---|
| **Who** | **Dev B** |
| **What** | Assembles final response JSON and sends to Unity |
| **Input (own data)** | `transcribed_text` (Step 3), `agent_audio_base64` (Step 10) |
| **Input (from Dev A)** | `reply_text`, `vendor_mood`, `negotiation_state` (Step 7) |
| **Output to Unity** | |
```json
{
  "session_id": "vr-session-abc123",
  "transcribed_text": "भाई ये silk scarf कितने का है?",
  "agent_reply_text": "अरे भाई, ये pure Banarasi silk है! ₹800 लगेगा",
  "agent_audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "vendor_mood": "enthusiastic",
  "negotiation_state": {
    "item": "silk_scarf",
    "quoted_price": 800,
    "vendor_happiness": 60,
    "vendor_patience": 68,
    "stage": "HAGGLING",
    "turn_count": 2,
    "deal_status": "negotiating"
  }
}
```

---

## Visual Summary — Who Touches the Data at Each Step

```
UNITY ──► audio_base64 + scene_scores JSON
              │
         ┌────▼─────┐
Step 1   │  Dev B    │  Receive & parse request
         └────┬──────┘
              │ audio_base64, language_code, scene_context
         ┌────▼─────┐
Step 2   │  Dev B    │  base64_to_bytes() → raw WAV bytes
         └────┬──────┘
              │ audio_bytes
         ┌────▼─────┐
Step 3   │  Dev B    │  transcribe_with_sarvam() → "भाई ये..."
         └────┬──────┘
              │ transcribed_text + scene_context metadata
         ┌────▼─────┐
Step 4   │  Dev B    │  memory.add_turn("user", text, metadata)
         └────┬──────┘
              │
    ┌─────────┴──────────┐  (parallel — asyncio.gather)
    │                    │
┌───▼────┐          ┌────▼───┐
│ Dev B  │ Step 5   │ Dev B  │ Step 6
│ context│          │ RAG    │
│ _block │          │ context│
└───┬────┘          └────┬───┘
    │                    │
    └─────────┬──────────┘
              │ context_block + rag_context + transcribed_text + scene_context
         ┌────▼─────┐
Step 7   │  Dev A    │  LLM prompt + Neo4j → vendor reply JSON
         │ (called  │  (Steps 7 + 7½ are internal to Dev A's function)
         │  by B)   │
         └────┬──────┘
              │ reply_text + new_mood + new_stage + price (validated)
         ┌────▼─────┐
Step 8   │  Dev B    │  memory.add_turn("vendor", reply, metadata)
         └────┬──────┘
              │ reply_text + language_code
         ┌────▼─────┐
Step 9   │  Dev B    │  speak_with_sarvam() → WAV audio bytes
         └────┬──────┘
              │ audio_bytes
         ┌────▼─────┐
Step 10  │  Dev B    │  bytes_to_base64() → base64 string
         └────┬──────┘
              │ all data assembled
         ┌────▼─────┐
Step 11  │  Dev B    │  Return InteractResponse JSON
         └────┬──────┘
              │
              ▼
           UNITY ◄── audio_base64 + reply JSON
```

---

## Dev B's Orchestrator Code (Conceptual)

This is what Dev B's endpoint handler looks like:

```python
# Dev B's main.py — /api/interact endpoint

@app.post("/api/interact")
async def interact(request: InteractRequest) -> InteractResponse:

    # Step 1 — Parse (Pydantic already validated)
    memory = get_memory(request.session_id)

    # Step 2 — Decode audio
    audio_bytes = base64_to_bytes(request.audio_base64)

    # Step 3 — Speech-to-Text
    transcribed_text = await transcribe_with_sarvam(audio_bytes, request.language_code)

    if transcribed_text == "":
        # Silence detected → short-circuit
        return build_silence_response(request.session_id)

    # Step 4 — Store user turn
    memory.add_turn("user", transcribed_text, extract_metadata(request.scene_context))

    # Steps 5 & 6 — Parallel: context history + RAG
    context_block, rag_context = await asyncio.gather(
        asyncio.to_thread(memory.get_context_block),
        retrieve_context(transcribed_text, n_results=3)
    )

    # Step 7 + 7½ — Call Dev A's brain (LLM + Neo4j validation)
    result = await generate_vendor_response(
        transcribed_text=transcribed_text,
        context_block=context_block,
        rag_context=rag_context,
        scene_context=request.scene_context,
        session_id=request.session_id
    )

    # Step 8 — Store vendor turn
    memory.add_turn("vendor", result["reply_text"], {
        "vendor_happiness": result["vendor_happiness"],
        "vendor_patience": result["vendor_patience"],
        "stage": result["new_stage"],
        "price": result["price_offered"]
    })

    # Step 9 — Text-to-Speech
    audio_bytes = await speak_with_sarvam(result["reply_text"], request.language_code)

    # Step 10 — Encode audio
    audio_base64 = bytes_to_base64(audio_bytes)

    # Step 11 — Return response
    return InteractResponse(
        session_id=request.session_id,
        transcribed_text=transcribed_text,
        agent_reply_text=result["reply_text"],
        agent_audio_base64=audio_base64,
        vendor_mood=result.get("vendor_mood", "neutral"),
        negotiation_state=build_negotiation_state(result)
    )
```

---

## Data Ownership Summary

| Data | Created By | Consumed By | Format |
|---|---|---|---|
| `audio_base64` (input) | Unity | Dev B (Step 2) | Base64 string |
| `audio_bytes` | Dev B (Step 2) | Dev B (Step 3) | Raw WAV bytes |
| `transcribed_text` | Dev B (Step 3) | Dev B (Steps 4, 6) + Dev A (Step 7 via arg) | Native script string |
| `scene_context` / scores JSON | Unity | Dev B (Steps 1, 4) + Dev A (Step 7 via arg) | Dict with happiness, patience, stage, items |
| `context_block` | Dev B (Step 5) | Dev A (Step 7 via arg — LLM prompt) | Multi-line text string |
| `rag_context` | Dev B (Step 6) | Dev A (Step 7 via arg — LLM prompt) | Multi-line text string |
| `reply_text` | Dev A (Step 7) | Dev B (Steps 8, 9, 11) | Native script string |
| `new_mood / stage / price` | Dev A (Step 7, validated via 7½) | Dev B (Steps 8, 11) | Dict |
| `audio_base64` (output) | Dev B (Step 10) | Dev B (Step 11) → Unity | Base64 string |

---

## Error Flow — What Happens When Things Break

```
Step 1 fails (bad request body)  → Dev B returns 422 to Unity (Pydantic validation)
Step 2 fails (bad base64)        → Dev B returns 400 to Unity immediately
Step 3 fails (Sarvam STT down)   → Dev B returns 503 to Unity
Step 3 returns "" (silence)      → Dev B skips Steps 4-6, vendor says "Kuch bola?"
Step 6 fails (ChromaDB down)     → Dev B sets rag_context="" and continues (graceful)
Step 7 fails (LLM error)         → Dev B returns 500 to Unity
Step 9 fails (Sarvam TTS down)   → Dev B sends text-only response (audio_base64="")
No SARVAM_API_KEY set            → Dev B auto-mocks STT/TTS, pipeline works with fake data
```

---

## Interface Contract: What Dev B Calls from Dev A

Dev A exposes **one function** that Dev B calls:

```python
# Dev A provides this (imported by Dev B)
async def generate_vendor_response(
    transcribed_text: str,
    context_block: str,
    rag_context: str,
    scene_context: dict,
    session_id: str
) -> dict:
    """
    Returns:
    {
        "reply_text": str,          # Vendor's spoken response
        "new_mood": int,            # Validated mood (0-100, clamped ±15)
        "new_stage": str,           # "GREETING"|"BROWSING"|"HAGGLING"|"DEAL"|"WALKAWAY"|"CLOSURE"
        "price_offered": int,       # Vendor's current asking price
        "vendor_happiness": int,    # 0-100
        "vendor_patience": int,     # 0-100
        "vendor_mood": str          # "enthusiastic"|"neutral"|"annoyed"|"angry"
    }
    """
```

---

*Document Version: 3.0 — Feb 13, 2026*
*Author: Developer B*
*Change: Dev B now owns the entry point (API endpoint) and exit point (response assembly)*
