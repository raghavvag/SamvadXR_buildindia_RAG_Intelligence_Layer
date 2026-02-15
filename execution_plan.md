# Samvad XR — Developer B: "Senses, Memory & Orchestration" Engine
## Execution Plan & Architecture Document (v3.0 — Dev B Owns Pipeline)

---

## 0. What Changed

| Area | v1.0 | v2.0 | v3.0 (Current) |
|---|---|---|---|
| Folder layout | Flat files at root | `services/` subfolder | Same |
| Async model | Sync `requests` | `async def` + `httpx.AsyncClient` | Same |
| Error types | Generic exceptions | Custom `SarvamServiceError`, `RAGServiceError` | Same |
| Memory ownership | We manage sessions | Dev A manages session dict | **Dev B manages session dict** (we own the endpoint) |
| **Pipeline ownership** | Not defined | **Dev A orchestrates**, calls our functions | **Dev B orchestrates** — we own entry + exit point |
| **API endpoint** | Not defined | Dev A owns `main.py` + `/api/interact` | **Dev B owns `main.py`** + `/api/interact` |
| State engine | Not addressed | Neo4j owned by Dev A | Same — Dev A exposes `generate_vendor_response()` |
| Language codes | 4 codes | 5 codes (added `hi-EN`) | Same |
| `get_context_block()` | Included metadata | Text-only output | Same |
| TTS audio format | Unspecified | WAV PCM 22kHz | Same |

---

## 1. High-Level Architecture (Revised)

```
Unity (C#)                         FastAPI Backend (Dev B's Orchestration)
┌──────────┐    POST /api/interact  ┌─────────────────────────────────────────────┐
│  VR User │ ───────────────────►   │  main.py (DEV B)                            │
│  in the  │                        │       │                                     │
│  Bazaar  │                        │  Step 1  Parse request              (Dev B) │
│          │                        │       ▼                                     │
│          │                        │  Step 2  base64_to_bytes            (Dev B) │
│          │                        │       ▼                                     │
│          │                        │  Step 3  transcribe_with_sarvam    (Dev B) │
│          │                        │       ▼                                     │
│          │                        │  Step 4  memory.add_turn("user")   (Dev B) │
│          │                        │       ▼                                     │
│          │                        │  Step 5  memory.get_context_block  (Dev B) │
│          │                        │       ▼                                     │
│          │                        │  Step 6  retrieve_context (RAG)    (Dev B) │
│          │                        │       ▼                                     │
│          │                        │  Step 7  generate_vendor_response  (Dev A) │
│          │                        │       │  ← Dev B calls Dev A here           │
│          │                        │       │  (LLM + Neo4j + validation)         │
│          │                        │       ▼                                     │
│          │                        │  Step 8  memory.add_turn("vendor") (Dev B) │
│          │                        │       ▼                                     │
│          │                        │  Step 9  speak_with_sarvam         (Dev B) │
│          │                        │       ▼                                     │
│          │                        │  Step 10 bytes_to_base64           (Dev B) │
│          │                        │       ▼                                     │
│          │  ◄──────────────────   │  Step 11 Return response           (Dev B) │
└──────────┘                        └─────────────────────────────────────────────┘
                                                      │
                                                      │ Step 7 (internal)
                                                      ▼
                                          ┌──────────────────────┐
                                          │  Dev A's Domain      │
                                          │                      │
                                          │  LLM Agent           │
                                          │  (ai_brain.py)       │
                                          │       +              │
                                          │  Neo4j State Graph   │
                                          │  (state_engine.py)   │
                                          │                      │
                                          │  Exposed as:         │
                                          │  generate_vendor_    │
                                          │  response()          │
                                          └──────────────────────┘
```

**We power steps 1–6, 8–11 (10 of 11 steps). We own the full pipeline and call Dev A for step 7 (LLM + state).**

---

## 2. File Structure (Aligned with Dev A's Mono-Repo)

```
SamVadXR-Orchestration/              ← Shared repo
│
├── main.py                           ← DEV B's entry point (FastAPI app + endpoint)
├── models.py                         ← DEV B's Pydantic schemas (InteractRequest, InteractResponse)
│
├── brain/                            ← DEV A's domain
│   ├── __init__.py
│   ├── ai_brain.py                   # LLM prompt + parsing
│   ├── state_engine.py               # Neo4j state machine
│   ├── prompts/                      # System prompt templates
│   └── config.py
│   # Dev A exposes: generate_vendor_response()
│
├── services/                         ← DEV B's domain (Senses + Memory)
│   ├── __init__.py
│   ├── middleware.py                  # base64_to_bytes, bytes_to_base64
│   ├── voice_ops.py                  # transcribe_with_sarvam, speak_with_sarvam
│   ├── rag_ops.py                    # initialize_knowledge_base, retrieve_context
│   ├── context_memory.py             # ConversationMemory class
│   └── exceptions.py                 # SarvamServiceError, RAGServiceError
│
├── data/                             ← Knowledge base text files (Dev B seeds)
│   ├── bargaining_culture.txt
│   ├── bazaar_items.txt
│   └── hindi_phrases.txt
│
├── tests/
│   ├── test_middleware.py            ← Dev B
│   ├── test_voice_ops.py            ← Dev B
│   ├── test_rag_ops.py              ← Dev B
│   ├── test_context_memory.py       ← Dev B
│   ├── test_api.py                  ← Dev B (endpoint tests)
│   └── test_integration.py          ← Joint
│
├── requirements.txt
└── .env.example
```

---

## 3. Module-by-Module Spec with Expected I/O

---

### 3.1 `services/exceptions.py` — Shared Error Types

Dev B catches these in the endpoint handler. Dev A may also raise them from `generate_vendor_response()`.

```python
class SarvamServiceError(Exception):
    """Raised when Sarvam AI API (STT or TTS) is unreachable or returns non-200."""
    def __init__(self, service: str, status_code: int = None, detail: str = ""):
        self.service = service        # "STT" or "TTS"
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Sarvam {service} error ({status_code}): {detail}")

class RAGServiceError(Exception):
    """Raised when ChromaDB query fails."""
    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(f"RAG error: {detail}")
```

**Dev B's usage (in main.py):**
```python
from services.exceptions import SarvamServiceError, RAGServiceError

try:
    text = await transcribe_with_sarvam(audio, "hi-IN")
except SarvamServiceError:
    # Return 503 to Unity: "Voice recognition unavailable"
    ...
```

---

### 3.2 `services/middleware.py` — The Codec Layer

**Sync functions.** Pure computation, no I/O.

| Function | Signature | Input | Output |
|---|---|---|---|
| `base64_to_bytes` | `def base64_to_bytes(b64_string: str) -> bytes` | `"data:audio/wav;base64,UklGRi..."` or plain `"UklGRi..."` | `b'\x52\x49\x46\x46...'` (raw WAV bytes) |
| `bytes_to_base64` | `def bytes_to_base64(audio_bytes: bytes) -> str` | `b'\x52\x49\x46\x46...'` | `"UklGRi..."` (plain Base64 string, no header) |

#### Error Contract
- `base64_to_bytes`: Raises `ValueError` with message `"Invalid base64 input: {detail}"` on malformed input.
- `bytes_to_base64`: Never fails on valid bytes.

#### Example
```python
# Strip data-URI header automatically
base64_to_bytes("data:audio/wav;base64,UklGRi...")  # → b'RIFF...'
base64_to_bytes("UklGRi...")                         # → b'RIFF...'  (also works)

bytes_to_base64(b'RIFF...')                          # → "UklGRi..."
```

---

### 3.3 `services/voice_ops.py` — Sarvam AI Voice Bridge

**Both functions are `async def`.** Uses `httpx.AsyncClient` for non-blocking I/O.

#### Function Signatures (Exact — Dev A is coding against these)

```python
async def transcribe_with_sarvam(audio_bytes: bytes, language_code: str) -> str: ...
async def speak_with_sarvam(text: str, language_code: str) -> bytes: ...
```

#### STT: `transcribe_with_sarvam`

| Property | Value |
|---|---|
| **Input** | Raw WAV bytes, language code (`"hi-IN"`, `"kn-IN"`, `"ta-IN"`, `"en-IN"`, `"hi-EN"`) |
| **Output** | Transcribed string in native script, e.g. `"भाई ये silk scarf कितने का है?"` |
| **On silence/noise** | Returns empty string `""` (Dev A's handler makes vendor say "Kuch bola aapne?") |
| **On API failure** | Raises `SarvamServiceError(service="STT", status_code=..., detail=...)` |
| **On missing API key (mock mode)** | Returns `"[MOCK] User said something in {language_code}"` |
| **Retries** | 1 automatic retry with 500ms backoff internally; raises after 2nd failure |
| **Latency budget** | ~800ms; Dev A's timeout: 5s |

#### Sarvam STT — Wire Format

```
POST https://api.sarvam.ai/speech-to-text
Headers:
  api-subscription-key: <SARVAM_API_KEY>

Body: multipart/form-data
  file: <audio.wav>   (filename="audio.wav", content_type="audio/wav")
  language_code: "hi-IN"
  model: "saarika:v2"

Response 200:
{
  "transcript": "भाई ये कितने का है?"
}
```

#### TTS: `speak_with_sarvam`

| Property | Value |
|---|---|
| **Input** | Text string in native script, language code |
| **Output** | Raw audio bytes (WAV PCM, 22kHz, mono) |
| **Audio format** | **WAV 16-bit PCM** |
| **Sample rate** | **22050 Hz (22kHz)** |
| **On API failure** | Raises `SarvamServiceError(service="TTS", status_code=..., detail=...)` |
| **On missing API key (mock mode)** | Returns 1-second silent WAV byte array (valid WAV header + zero samples) |
| **Retries** | 1 automatic retry with 500ms backoff |
| **Latency budget** | ~600ms; Dev A's timeout: 5s |

#### Sarvam TTS — Wire Format

```
POST https://api.sarvam.ai/text-to-speech
Headers:
  api-subscription-key: <SARVAM_API_KEY>
  Content-Type: application/json

Body:
{
  "inputs": ["सौ रुपये, बस"],
  "target_language_code": "hi-IN",
  "speaker": "meera",
  "model": "bulbul:v1"
}

Response 200:
{
  "audios": ["<base64 encoded audio string>"]
}
```

**Note:** Sarvam TTS returns Base64 audio in the JSON response. Our `speak_with_sarvam` decodes this internally and returns raw bytes to Dev A.

#### Supported Language Codes (Standardized — agreed with Dev A)

| Language | Code | Notes |
|---|---|---|
| Hindi | `hi-IN` | Primary |
| English (India) | `en-IN` | Fallback |
| Hinglish | `hi-EN` | Mixed-code speech |
| Kannada | `kn-IN` | Phase 2 |
| Tamil | `ta-IN` | Phase 2 |

---

### 3.4 `services/context_memory.py` — Conversation History Manager

**All methods are sync** (in-memory operations only). Dev A instantiates one per session on his side.

#### Design Constraints

1. **Instance-based** — not a singleton. Dev B creates one `ConversationMemory()` per session in `main.py`.
2. **No required constructor args** — `ConversationMemory()` works, `ConversationMemory(session_id="abc")` also works.
3. **Arbitrary metadata dict** — stored as-is. We pass things like `{"held_item": "silk_scarf", "mood": 55, "stage": "BROWSING"}`.
4. **`get_context_block()` returns text-only** — no metadata in the output. Dev A injects mood/stage separately from his Neo4j state engine.
5. **Window size configurable** — default 10 turns.

#### Data Model

```python
@dataclass
class DialogueTurn:
    role: str              # "user" | "vendor"
    text: str              # The spoken text (native script)
    timestamp: float       # time.time() at creation
    metadata: dict         # Arbitrary dict, stored as-is
```

#### Class Signature

```python
class ConversationMemory:
    def __init__(self, session_id: str = "", max_window: int = 10): ...
    def add_turn(self, role: str, text: str, metadata: dict = None) -> None: ...
    def get_context_block(self) -> str: ...
    def get_recent_turns(self, n: int = 5) -> list[DialogueTurn]: ...
    def clear(self) -> None: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory": ...
```

#### How Dev B Orchestrates It (From Our main.py)

```python
# Dev B's main.py — we own the endpoint
from services.context_memory import ConversationMemory
from services.middleware import base64_to_bytes, bytes_to_base64
from services.voice_ops import transcribe_with_sarvam, speak_with_sarvam
from services.rag_ops import retrieve_context
from brain import generate_vendor_response  # ← Dev A's function

sessions = {}  # session_id → ConversationMemory (Dev B manages this)

def get_memory(session_id: str) -> ConversationMemory:
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory()
    return sessions[session_id]

# Inside the /api/interact handler:
memory = get_memory(request.session_id)

# Steps 2-3: Decode + STT
audio_bytes = base64_to_bytes(request.audio_base64)
text = await transcribe_with_sarvam(audio_bytes, request.language_code)

# Step 4: Store user turn
memory.add_turn("user", text, {
    "held_item": "silk_scarf",
    "looked_at_item": "brass_statue",
    "mood": 55,
    "stage": "BROWSING"
})

# Steps 5-6: Context + RAG (parallel)
context_block, rag_ctx = await asyncio.gather(
    asyncio.to_thread(memory.get_context_block),
    retrieve_context(text, n_results=3)
)

# Step 7: Call Dev A's brain
result = await generate_vendor_response(
    transcribed_text=text,
    context_block=context_block,
    rag_context=rag_ctx,
    scene_context=request.scene_context,
    session_id=request.session_id
)

# Step 8: Store vendor turn
memory.add_turn("vendor", result["reply_text"], {
    "mood": result["new_mood"],
    "stage": result["new_stage"],
    "price": result["price_offered"]
})

# Steps 9-10: TTS + Encode
audio = await speak_with_sarvam(result["reply_text"], request.language_code)
audio_b64 = bytes_to_base64(audio)

# Step 11: Return response to Unity
return InteractResponse(...)
```

#### `get_context_block()` — Output Format (TEXT-ONLY, per Dev A's request)

```
[Summary of earlier conversation]
The user greeted the vendor and asked about silk scarves. The vendor quoted ₹800.
The user expressed shock and counter-offered ₹300.

[Recent Dialogue]
[Turn 4] User: चलो ₹400 में दे दो
[Turn 4] Vendor: ₹400? भाई, इतने में तो धागा भी नहीं आता!
[Turn 5] User: ठीक है, ₹450 final
[Turn 5] Vendor: ₹500 से नीचे impossible है
[Turn 6] User: ok ₹500 done
```

**No metadata, no mood, no stage** — Dev A adds those from his Neo4j state engine output.

#### Memory Compression Strategy

```
Window = 10 turns (configurable)

Turns 1–10:  Stored verbatim in self._turns[]
Turn 11:     Oldest 5 turns → compressed into self._rolling_summary (string concat)
             Turns 6–11 remain verbatim
Turn 20:     Oldest batch again → appended to self._rolling_summary
             Recent window remains verbatim
```

**Hackathon:** Simple string concatenation of `"[role]: text"` lines.
**Production:** LLM-powered summarization call (async, off critical path).

#### Interaction with Neo4j State Engine (Dev A's Domain)

We do **NOT** talk to Neo4j directly. The flow:

```
                    Dev B's Domain                    Dev A's Domain
                 ┌──────────────────┐           ┌─────────────────────┐
                 │ ConversationMemory│           │ Neo4j State Graph   │
                 │                  │           │                     │
 add_turn() ───►│ Stores text +    │           │ Stores:             │
                 │ metadata as-is   │           │  - Current stage    │
                 │                  │           │  - Mood (0-100)     │
                 │ get_context_block│───────►   │  - Price state      │
                 │ (text history)   │  Dev A    │  - Valid transitions│
                 │                  │  merges   │                     │
                 │                  │  both     │ Validates:          │
                 │                  │  into     │  - Stage transitions│
                 │                  │  LLM      │  - Mood ±15 clamp  │
                 │                  │  prompt   │  - Legal actions    │
                 └──────────────────┘           └─────────────────────┘
```

**Key boundary:** We store _conversational text history_. Dev A stores _game state_ in Neo4j. The LLM prompt sees both — Dev B passes our `context_block` to Dev A's `generate_vendor_response()`, which merges it with Neo4j state.

#### Metadata We Expect to Receive (Stored As-Is)

| Key | Type | Example | Source |
|---|---|---|---|
| `held_item` | `str` | `"silk_scarf"` | Unity scene_context |
| `looked_at_item` | `str` | `"brass_statue"` | Unity scene_context |
| `mood` | `int` | `55` | Neo4j state engine |
| `stage` | `str` | `"BROWSING"` | Neo4j state engine |
| `price` | `int` | `700` | LLM output (validated by Dev A) |

We don't validate or interpret these keys. They are opaque data stored for potential future use (analytics, session replay).

---

### 3.5 `services/rag_ops.py` — Cultural Knowledge Retrieval

**`retrieve_context` is `async def`** (wraps sync ChromaDB in `asyncio.to_thread()`).

**`initialize_knowledge_base` is sync** (called once at app startup by Dev B in FastAPI lifespan).

#### Function Signatures

```python
def initialize_knowledge_base() -> None: ...
async def retrieve_context(query: str, n_results: int = 3) -> str: ...
```

#### Behavior Contract

| Scenario | Return Value |
|---|---|
| Normal results found | Single concatenated string of top N chunks, separated by `\n` |
| No relevant results | Empty string `""` |
| ChromaDB unreachable/error | Raises `RAGServiceError(detail="...")` |

#### ChromaDB Setup

```python
import chromadb

# In-process, in-memory client — no external service to deploy
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="samvad_context",
    metadata={"hnsw:space": "cosine"}
)
```

#### Ingestion Pipeline

```
data/*.txt
    │
    ▼
Read each file → Split into ~200-word chunks
    │
    ▼
collection.add(
    documents=[chunk1, chunk2, ...],
    ids=["bargaining_0", "bargaining_1", ...],
    metadatas=[{"source": "bargaining_culture.txt"}, ...]
)
    │
    ▼
ChromaDB auto-embeds using default sentence-transformers model
```

#### Retrieval Example

```python
result = await retrieve_context("user wants to buy silk scarf, what is fair price?", n_results=2)
# Returns (single string):
# "Silk Scarf: Wholesale ₹150, Fair Retail ₹300-400, Tourist Price ₹800-1200\n
#  In Indian street markets, the vendor's first price is always 2-4 times the
#  actual value. A good buyer starts at 25-30% of the quoted price."
```

#### Expected Data Files

**`data/bargaining_culture.txt`**
```
In Indian street markets (bazaars), bargaining is not just expected—it is a social ritual.
The vendor's first price is always 2-4 times the actual value. A good buyer starts at
25-30% of the quoted price and works upward. Showing too much interest in an item
raises its price. A common strategy is to pick up a cheaper item first, pretend to be
interested in it, then "casually" notice the real target. Walking away is the most
powerful bargaining lever—the vendor will often call you back with a lower price.
Touching or picking up an item signals commitment—the vendor may refuse to let you
leave without a deal. Eye contact and friendly banter build rapport and can lower prices.
```

**`data/bazaar_items.txt`**
```
Silk Scarf: Wholesale ₹150, Fair Retail ₹300-400, Tourist Price ₹800-1200. Banarasi silk is the most famous.
Brass Keychain: Wholesale ₹20, Fair Retail ₹50, Tourist Price ₹150. Common souvenir, low margins.
Spice Box (Masala Dabba): Wholesale ₹200, Fair Retail ₹400, Tourist Price ₹900. Stainless steel, 7 compartments.
Leather Wallet: Wholesale ₹100, Fair Retail ₹250, Tourist Price ₹600. Genuine leather vs PU leather.
Pashmina Shawl: Wholesale ₹500, Fair Retail ₹1000-1500, Tourist Price ₹3000-5000. Many are fake.
```

**`data/hindi_phrases.txt`**
```
"Kitne ka hai?" (कितने का है?) — "How much is this?" — The universal opener.
"Bahut mehnga hai!" (बहुत महंगा है!) — "Too expensive!" — Show shock.
"Thoda kam karo" (थोड़ा कम करो) — "Reduce it a bit" — Polite negotiation.
"Chhod do yaar" (छोड़ दो यार) — "Forget it, man" — The walk-away bluff.
"Last price bolo" (लास्ट प्राइस बोलो) — "Tell me the final price" — You're serious.
"Bhaiya, student hoon" (भैया, स्टूडेंट हूँ) — "Brother, I'm a student" — Sympathy card.
"Sab jagah yahi milta hai" (सब जगह यही मिलता है) — "This is available everywhere" — Undermine uniqueness.
```

---

## 4. Data Flow — Complete Request Lifecycle (Revised)

```
Step  Owner   Module              Action                                         Time
────  ──────  ──────────────────  ─────────────────────────────────────────────   ────
 1    Dev B   main.py             Receive POST /api/interact, parse request        0ms
 2    Dev B   middleware.py        base64_to_bytes(request.audio_base64)           1ms
 3    Dev B   voice_ops.py         await transcribe_with_sarvam(bytes, "hi-IN")  ~800ms
                                   → "भाई ये silk scarf कितने का है?"
 4    Dev B   context_memory.py    memory.add_turn("user", text, metadata)         1ms
 5    Dev B   context_memory.py    context_block = memory.get_context_block()       1ms
 6    Dev B   rag_ops.py           rag_ctx = await retrieve_context(text, 3)      ~50ms
 7    Dev A   brain/ai_brain.py    Dev B calls generate_vendor_response()          ~2s
                                     (LLM prompt + Neo4j state + validation)
                                     Steps 7 + 7½ are internal to Dev A's function
 8    Dev B   context_memory.py    memory.add_turn("vendor", reply, metadata)       1ms
 9    Dev B   voice_ops.py         audio = await speak_with_sarvam(reply, "hi-IN") ~600ms
10    Dev B   middleware.py         b64 = bytes_to_base64(audio)                     1ms
11    Dev B   main.py              Return InteractResponse to Unity                 0ms
                                                                    TOTAL ≈ 3.5s
```

**Parallelization:** Dev B runs Steps 5 and 6 concurrently via `asyncio.gather()`.

---

## 5. Step-by-Step Implementation Plan (Dev B)

> Each step below is one atomic unit of work. Steps marked **🤝 Dev A** require coordination.
> Build and test each step before moving to the next.

---

### Phase 1 — Project Skeleton & Shared Contracts

**Goal:** Set up the repo structure and the shared exception types Dev A will import.

**Step 1.1 — Create folder structure and config files**
- Create `main.py` (FastAPI app skeleton with `/api/interact` endpoint)
- Create `models.py` (Pydantic schemas: `InteractRequest`, `InteractResponse`)
- Create `services/__init__.py` (empty, makes it a package)
- Create `requirements.txt` with all dependencies
- Create `.env.example` with placeholder keys
- Create `data/` folder (empty for now)
- Create `tests/` folder with `__init__.py`
```
Files created:
  main.py                  ← NEW: FastAPI app (Dev B owns this)
  models.py                ← NEW: Pydantic request/response schemas
  services/__init__.py
  requirements.txt
  .env.example
  data/                    (empty dir)
  tests/__init__.py
```

**Step 1.2 — Write `services/exceptions.py`**
- Define `SarvamServiceError(service, status_code, detail)`
- Define `RAGServiceError(detail)`
- These are the exact exception types Dev A catches in his orchestrator
```python
# What Dev A imports:
from services.exceptions import SarvamServiceError, RAGServiceError
```

**Step 1.3 — Write `services/middleware.py`**
- `def base64_to_bytes(b64_string: str) -> bytes`
  - Strip `data:audio/wav;base64,` header if present
  - Decode and return raw bytes
  - Raise `ValueError` on invalid input
- `def bytes_to_base64(audio_bytes: bytes) -> str`
  - Encode bytes to clean base64 string (no header prefix)
- Both are **sync** — pure computation, no I/O

**Step 1.4 — Write `tests/test_middleware.py`**
- Test normal base64 roundtrip
- Test with data-URI header stripping
- Test invalid input raises `ValueError`
- Run tests, confirm all pass

---

### Phase 2 — Voice Pipeline (Sarvam STT & TTS)

**Goal:** Build the "ears" and "mouth" — async functions that talk to Sarvam AI.

**Step 2.1 — Write `services/voice_ops.py` — STT function**
- `async def transcribe_with_sarvam(audio_bytes: bytes, language_code: str) -> str`
- Use `httpx.AsyncClient` for async HTTP
- POST to `https://api.sarvam.ai/speech-to-text` with multipart form data
- 1 internal retry with 500ms backoff on failure
- Mock mode: if no `SARVAM_API_KEY`, return `"[MOCK] User said something in {lang}"`
- On silence/noise: return `""`
- On failure after retry: raise `SarvamServiceError(service="STT", ...)`

**Step 2.2 — Write `services/voice_ops.py` — TTS function**
- `async def speak_with_sarvam(text: str, language_code: str) -> bytes`
- POST to `https://api.sarvam.ai/text-to-speech` with JSON body
- Sarvam returns base64 audio in JSON → we decode internally → return raw WAV bytes
- Mock mode: return 1-second silent WAV (valid WAV header + zero samples at 22kHz)
- On failure after retry: raise `SarvamServiceError(service="TTS", ...)`
- Helper: `_generate_silent_wav() -> bytes` for mock mode

**Step 2.3 — Write `tests/test_voice_ops.py`**
- Test mock mode returns expected shapes (string for STT, bytes for TTS)
- Test mock WAV is a valid WAV file (check RIFF header)
- Mock `httpx` transport to test real API call path without burning credits
- Test retry logic (first call fails, second succeeds)
- Test `SarvamServiceError` raised after both retries fail

**Step 2.4 — Integration test: middleware + voice roundtrip**
- base64 string → `base64_to_bytes()` → `transcribe_with_sarvam()` → text
- text → `speak_with_sarvam()` → audio bytes → `bytes_to_base64()` → base64 string
- Verify full loop runs without errors in mock mode

---

### Phase 3 — Knowledge Engine (ChromaDB RAG)

**Goal:** Build the cultural "cheat sheet" the LLM uses to know real prices and bargaining norms.

**Step 3.1 — Create seed data files**
- `data/bargaining_culture.txt` — Indian bazaar negotiation rules, social cues, walk-away strategies
- `data/bazaar_items.txt` — Item names with wholesale / retail / tourist prices
- `data/hindi_phrases.txt` — Common bargaining phrases in Hindi with transliterations and translations

**Step 3.2 — Write `services/rag_ops.py` — initialization**
- `def initialize_knowledge_base() -> None`
- Create `chromadb.Client()` (in-process, in-memory)
- Create/get collection `"samvad_context"` with cosine distance
- Read all `data/*.txt` files
- Chunk each file into ~200-word segments
- Add chunks to collection with source metadata and auto-generated IDs
- Called once by **Dev B** in the FastAPI lifespan startup event

**Step 3.3 — Write `services/rag_ops.py` — retrieval**
- `async def retrieve_context(query: str, n_results: int = 3) -> str`
- Wrap sync `collection.query()` in `asyncio.to_thread()`
- Join top N results with `\n` into a single string
- Return `""` if no results found
- Raise `RAGServiceError` if ChromaDB fails

**Step 3.4 — Write `tests/test_rag_ops.py`**
- Test: initialize, then query "silk scarf price" → result contains "₹" and "Silk"
- Test: query with no matching results → returns `""`
- Test: chunking produces expected number of chunks from test data

---

### Phase 4 — Context Memory Layer

**Goal:** Give the LLM memory of the ongoing conversation so the vendor doesn't "forget" mid-negotiation.

**Step 4.1 — Define data model in `services/context_memory.py`**
- `@dataclass DialogueTurn` with `role`, `text`, `timestamp`, `metadata`
- Role is `"user"` or `"vendor"`
- Metadata is an arbitrary dict (stored as-is, never interpreted by us)

**Step 4.2 — Implement `ConversationMemory.__init__` and `add_turn`**
- `def __init__(self, session_id: str = "", max_window: int = 10)`
  - No required args (Dev A can call `ConversationMemory()` bare)
  - Instance-based — no global state, safe for concurrent sessions
- `def add_turn(self, role: str, text: str, metadata: dict = None) -> None`
  - Append `DialogueTurn` to `self._turns[]`
  - If `len(self._turns) > max_window`: trigger `_summarize_overflow()`

**Step 4.3 — Implement `_summarize_overflow`**
- Take oldest half of turns beyond the window
- Concatenate their `"[role]: text"` lines into a summary paragraph
- Append to `self._rolling_summary` string
- Remove those turns from `self._turns[]`
- Hackathon version: simple string concatenation (no LLM call)

**Step 4.4 — Implement `get_context_block`**
- `def get_context_block(self) -> str`
- Returns **text-only** string (no metadata, no mood, no stage)
- **🤝 Dev A** injects mood/stage separately from his Neo4j state
- Format:
```
[Summary of earlier conversation]
{self._rolling_summary}

[Recent Dialogue]
[Turn 1] User: Namaste bhaiya!
[Turn 1] Vendor: Aao aao!
[Turn 2] User: भाई ये silk scarf कितने का है?
```

**Step 4.5 — Implement utility methods**
- `def get_recent_turns(self, n: int = 5) -> list[DialogueTurn]` — last N turns
- `def clear(self) -> None` — full wipe (turns + summary)
- `def to_dict(self) -> dict` — serialize for future Redis persistence
- `@classmethod from_dict(cls, data: dict) -> ConversationMemory` — deserialize

**Step 4.6 — Write `tests/test_context_memory.py`**
- Test: add 3 turns, `get_context_block()` returns all 3 formatted
- Test: add 12 turns (exceeds window=10), verify summary is created and recent turns are kept
- Test: `clear()` resets everything
- Test: `to_dict()` → `from_dict()` roundtrip preserves all data
- Test: metadata is stored as-is and accessible via `get_recent_turns()`
- Test: no-args constructor works (`ConversationMemory()`)

---

### Phase 5 — Wiring & Integration with Dev A

**Goal:** Connect Dev A's `generate_vendor_response()` to our pipeline and run the full loop.

**Step 5.1 — Set up `services/__init__.py` exports**
- Add clean exports so our `main.py` can do:
```python
from services.middleware import base64_to_bytes, bytes_to_base64
from services.voice_ops import transcribe_with_sarvam, speak_with_sarvam
from services.rag_ops import initialize_knowledge_base, retrieve_context
from services.context_memory import ConversationMemory
from services.exceptions import SarvamServiceError, RAGServiceError
```
- Run an import-only test to verify nothing crashes on import

**Step 5.2 — Write contract test: `tests/test_contract.py`**
- Verify every function signature matches what `main.py` expects
- Verify return types (str, bytes, None)
- Verify exception types are importable and catchable
- This is the "handshake test" — if it passes, our pipeline is wired correctly

**Step 5.3 — 🤝 Dev A: Provide `generate_vendor_response()` function**
- Dev A provides his function (or a mock/stub version of it)
- Dev B imports it in `main.py`: `from brain import generate_vendor_response`
- Verify it returns the expected dict shape: `{reply_text, new_mood, new_stage, price_offered, ...}`

**Step 5.4 — Dev B pipeline smoke test (mock everything)**
- Run full request cycle with mock Sarvam (no API key) + mock `generate_vendor_response`
- Verify: audio in → mock transcription → memory stored → RAG retrieved → mock LLM reply → mock TTS → audio out
- All within our `main.py` endpoint — no Dev A needed for this test

**Step 5.5 — 🤝 Dev A: Swap mock brain for real `generate_vendor_response()`**
- Replace mock with Dev A's real function
- Run through one full request cycle
- Verify Dev A's Neo4j state validation works with our context_block/rag_context

**Step 5.6 — 🤝 Dev A: Full end-to-end with real Sarvam API**
- Set `SARVAM_API_KEY` in env
- Send real audio → Sarvam STT → transcribed Hindi text
- Store in memory → retrieve RAG context
- **Call Dev A's `generate_vendor_response()`** (real LLM + Neo4j)
- Store vendor reply in memory
- Sarvam TTS → audio back
- Measure total latency (target: < 4s)

---

### Implementation Order Visualization

```
Week 1:
  ┌─────────────────────────────────────────────────┐
  │ Phase 1: Skeleton + Middleware (Dev B solo)      │
  │ Steps 1.1 → 1.4                                 │
  └──────────────────────┬──────────────────────────┘
                         ▼
  ┌─────────────────────────────────────────────────┐
  │ Phase 2: Voice Pipeline (Dev B solo)             │
  │ Steps 2.1 → 2.4                                 │
  │                                                  │
  │ Meanwhile Dev A builds:                          │
  │   - LLM agent with mock STT/TTS/RAG             │
  │   - Neo4j state engine                           │
  │   - FastAPI orchestration                        │
  └──────────────────────┬──────────────────────────┘
                         ▼
  ┌─────────────────────────────────────────────────┐
  │ Phase 3: RAG Knowledge Engine (Dev B solo)       │
  │ Steps 3.1 → 3.4                                 │
  └──────────────────────┬──────────────────────────┘
                         ▼
  ┌─────────────────────────────────────────────────┐
  │ Phase 4: Context Memory (Dev B solo)             │
  │ Steps 4.1 → 4.6                                 │
  └──────────────────────┬──────────────────────────┘
                         ▼
Week 2:
  ┌─────────────────────────────────────────────────┐
  │ Phase 5: Integration (Dev A + Dev B together)    │
  │ Steps 5.1 → 5.6                                 │
  │                                                  │
  │ 🤝 Dev A provides generate_vendor_response()     │
  │ 🤝 Joint E2E testing                            │
  │ 🤝 Latency tuning                               │
  └─────────────────────────────────────────────────┘
```

---

## 7. Error Handling Contract (Full Table)

| Scenario | Our Function | What We Do | What Dev B's Endpoint Does |
|----------|-------------|-----------|----------------|
| Sarvam STT API down | `transcribe_with_sarvam` | Retry 1x (500ms delay), then raise `SarvamServiceError(service="STT")` | Returns 503 to Unity: "Voice recognition unavailable" |
| Audio is silence/noise | `transcribe_with_sarvam` | Return empty string `""` | Vendor says "Kuch bola aapne?" |
| STT returns garbled text | `transcribe_with_sarvam` | Return whatever Sarvam returns (best effort) | LLM brain handles garbled input gracefully |
| Sarvam TTS API down | `speak_with_sarvam` | Retry 1x (500ms delay), then raise `SarvamServiceError(service="TTS")` | Returns text-only response (subtitle mode) |
| No SARVAM_API_KEY | both voice functions | Return mock data (no exception raised) | Pipeline works end-to-end with fake data |
| ChromaDB no results | `retrieve_context` | Return empty string `""` | Dev A's LLM proceeds without cultural context |
| ChromaDB crashes | `retrieve_context` | Raise `RAGServiceError(detail="...")` | Skips RAG, continues (graceful degradation) |
| Invalid base64 input | `base64_to_bytes` | Raise `ValueError` | Returns 400 to Unity: "Invalid audio data" |
| LLM error | Dev A's `generate_vendor_response` | Dev A raises exception | Returns 500 to Unity |

---

## 8. Environment Variables

```env
# .env.example
SARVAM_API_KEY=your_sarvam_api_key_here
LLM_API_KEY=your_llm_api_key_here              # Dev A's domain
NEO4J_URI=bolt://localhost:7687                  # Dev A's domain
NEO4J_USER=neo4j                                 # Dev A's domain
NEO4J_PASSWORD=password                          # Dev A's domain
LOG_LEVEL=INFO
MOCK_MODE=false                                  # Auto-true if SARVAM_API_KEY missing
```

---

## 9. Dependencies

```
# requirements.txt
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0
httpx>=0.25.0              # Async HTTP client (replaces requests)
chromadb>=0.4.0
python-dotenv>=1.0.0
python-multipart>=0.0.6    # For multipart file upload to Sarvam
pytest>=7.4.0
pytest-asyncio>=0.23.0     # For testing async functions
```

---

## 10. Risk Register & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Sarvam API downtime | Voice pipeline blocked | Mock mode auto-activates; 1 retry with 500ms backoff |
| Latency > 5s per turn | Breaks VR immersion | `asyncio.gather()` for parallel steps 5+6; TTS cache for repeated phrases |
| ChromaDB embedding quality | Bad RAG retrieval | Carefully curated seed data; manual testing with real queries |
| Conversation memory bloat | High token cost, slow LLM | Sliding window (10 turns) + summary compression caps at ~500 tokens |
| Hinglish confuses STT | Incorrect transcription | Sarvam trained on Hinglish; `hi-EN` language code supported |
| Neo4j state ↔ memory desync | Vendor contradicts game state | We store metadata as-is; Dev A is single source of truth |
| Import path mismatch | Integration fails at swap | Phase 5 step 5.1 — verify `from services.X import Y` works |

---

## 11. Testing Strategy

| Test Type | What | Owner | Tool |
|---|---|---|---|
| **Unit** | Each function in isolation (mocked Sarvam, mocked ChromaDB) | Dev B | `pytest` + `pytest-asyncio` |
| **Integration** | Full pipeline: Base64 → STT → Memory → RAG → (mock LLM) → TTS → Base64 | Dev B | `pytest` + `httpx.AsyncClient` |
| **Contract** | Verify our function signatures match Dev A's mock interfaces exactly | Joint | Import validation test |
| **E2E** | Real audio → full pipeline via Dev B's endpoint | Dev B | `curl` + browser |
| **VR E2E** | Unity sends real request, receives audio, plays in headset | Joint | Unity build |

---

## 12. Success Criteria (Hackathon Demo Checklist)

- [ ] User speaks Hindi into VR headset mic
- [ ] Backend transcribes speech correctly (Sarvam STT)
- [ ] Agent remembers previous turns (context memory window)
- [ ] Agent uses cultural knowledge (RAG) for realistic prices
- [ ] Neo4j state engine tracks mood and stage correctly (Dev A)
- [ ] Agent responds in Hindi with appropriate accent (Sarvam TTS)
- [ ] Response latency < 4 seconds end-to-end
- [ ] Vendor mood visibly changes based on negotiation tactics
- [ ] "Walk away" strategy triggers vendor callback at lower price
- [ ] Full 5+ turn negotiation works without coherence breaks
- [ ] Mock mode allows full testing without burning Sarvam credits
- [ ] `USE_MOCKS=false` swap works with zero code changes in Dev B's pipeline

---

*Document Version: 3.0 — Revised Feb 13, 2026 — Dev B Owns Pipeline Orchestration*
*Author: Developer B — "Senses, Memory & Orchestration" Engineer*
