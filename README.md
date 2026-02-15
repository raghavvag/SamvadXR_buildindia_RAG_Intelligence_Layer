# SamvadXR RAG Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/Sarvam_AI-FF4081?style=for-the-badge&logo=ai&logoColor=white" />
  <img src="https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white" />
  <img src="https://img.shields.io/badge/Unity-000000?style=for-the-badge&logo=unity&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Uvicorn-2D3748?style=for-the-badge&logo=gunicorn&logoColor=white" />
  <img src="https://img.shields.io/badge/httpx-4B8BBE?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Flow](#pipeline-flow)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [API Endpoints](#api-endpoints)
- [Request and Response Schemas](#request-and-response-schemas)
- [Services](#services)
- [Session Management](#session-management)
- [RAG Pipeline](#rag-pipeline)
- [Language Processing Pipeline](#language-processing-pipeline)
- [Environment Variables](#environment-variables)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)
- [Development Notes](#development-notes)

---

## Overview

SamvadXR RAG Pipeline is the intelligence engine behind SamvadXR, a VR-based language immersion platform where users practice real-world conversational skills (bargaining, negotiation) with AI-powered vendors in virtual Indian marketplaces.

The system orchestrates a multi-step pipeline that converts spoken audio from a Unity VR frontend into contextual, multilingual vendor responses with full audio output. It combines Retrieval-Augmented Generation (RAG), conversational memory, multilingual speech processing, and a graph-backed LLM brain to deliver natural, culturally aware interactions.

### Key Capabilities

- Real-time Speech-to-Text and Text-to-Speech in 10+ Indian languages via Sarvam AI
- Retrieval-Augmented Generation using ChromaDB for vendor knowledge grounding
- Conversational memory with sliding-window context management
- Transliteration of native script responses to romanized Latin text
- English caption generation for all vendor and suggested responses
- Session-aware negotiation state tracking
- Fallback mechanisms at every pipeline stage

---

## Architecture

```
+-------------------+          +---------------------------+          +-------------------+
|                   |  Audio   |                           |  HTTP    |                   |
|   Unity VR        |  (b64)   |   Dev B (This Repo)       |  JSON    |   Dev A (Brain)   |
|   Frontend        +--------->+   FastAPI Orchestrator     +--------->+   LLM + Neo4j     |
|                   |          |                           |          |   Knowledge Graph  |
|   - C# Client     |<---------+   Pipeline Steps 1-11     |<---------+                   |
|   - Quest 3       |  JSON +  |                           |  JSON    |   - Gemini LLM     |
|                   |  Audio   |                           |          |   - Neo4j Graph    |
+-------------------+          +-------------+-------------+          +-------------------+
                                             |
                               +-------------+-------------+
                               |                           |
                    +----------v----------+     +----------v----------+
                    |                     |     |                     |
                    |   Sarvam AI APIs    |     |   ChromaDB          |
                    |                     |     |   (Vector Store)    |
                    |   - STT (Speech)    |     |                     |
                    |   - TTS (Voice)     |     |   - Seed Knowledge  |
                    |   - Translate       |     |   - Conversation    |
                    |   - Transliterate   |     |     History         |
                    |   - Lang Detect     |     |                     |
                    +---------------------+     +---------------------+
```

### Two-Developer Architecture

| Role | Responsibility | Communication |
|------|----------------|---------------|
| **Dev B (This Repo)** | Pipeline orchestration, STT/TTS, translation, transliteration, RAG, session management, Unity API contract | Receives audio from Unity, returns JSON + audio |
| **Dev A (External)** | LLM reasoning, Neo4j knowledge graph, vendor personality, negotiation logic, price generation | Receives English text + context, returns vendor reply JSON |

---

## Pipeline Flow

The system executes an 11-step pipeline for every user interaction:

```
Step 1   [Unity]        User speaks into VR headset microphone
                        |
Step 2   [Dev B]        Decode base64 audio --> raw WAV bytes
                        |
Step 3   [Sarvam AI]    Speech-to-Text (STT) in target language
                        |
Step 3.5A [Dev B]       Detect actual spoken language (script analysis + API)
                        |
Step 3.5C [Sarvam AI]   Translate user's text --> English (for Dev A)
                        |
Step 4   [Dev B]        Store user turn in ConversationMemory
                        |
Step 5   [Dev B]        Build context block from conversation history
Step 6   [ChromaDB]     RAG retrieval: semantic search for relevant knowledge
                        |  (Steps 5 & 6 run in parallel)
                        |
Step 7   [Dev A]        Generate vendor response via LLM + Neo4j
                        |  Input:  English text + context + RAG + scene state
                        |  Output: reply_text, negotiation_state, happiness, mood
                        |
Step 8   [Dev B]        Store vendor turn in ConversationMemory
Step 8.1 [ChromaDB]     Index conversation turn into RAG for future retrieval
                        |
Step 8.5 [Sarvam AI]    Translate vendor reply: English --> target language (native script)
                        |
Step 8.6 [Sarvam AI]    Transliterate: native script --> romanized Latin text
                        |
Step 9   [Sarvam AI]    Text-to-Speech (TTS) using native script text
Step 10  [Dev B]        Encode TTS audio --> base64
                        |  (Audio uses native script; text uses romanized)
                        |
Step 11  [Dev B]        Assemble final response JSON
Step 11.1 [Sarvam AI]   Translate + transliterate suggested_user_response
Step 11.2 [Sarvam AI]   Generate English captions for reply + suggested
                        |
         [Unity]        Display romanized text + English captions + play audio
```

---

## Project Structure

```
SamvadXR_intelligence_engine/
|
|-- main.py                          # FastAPI application entry point (Dev B)
|-- models.py                        # Pydantic request/response schemas
|-- requirements.txt                 # Python dependencies
|-- .env                             # Environment variables (API keys, URLs)
|
|-- services/
|   |-- voice_ops.py                 # Sarvam AI integrations (STT, TTS, translate, transliterate, detect)
|   |-- rag_ops.py                   # ChromaDB RAG operations (init, retrieve, add, clear)
|   |-- context_memory.py            # ConversationMemory with sliding window
|   |-- middleware.py                # Base64 encode/decode utilities
|   |-- exceptions.py               # Custom exception classes
|
|-- data/
|   |-- vendor_knowledge.txt         # Seed knowledge for RAG (vegetable market data)
|   |-- (other .txt files)           # Additional domain knowledge files
|
|-- venv/                            # Python virtual environment
|
|-- docs/
|   |-- execution_plan.md            # Detailed pipeline execution plan
|   |-- api_contract.md              # Unity <-> Backend API contract
|   |-- architecture.md              # System architecture documentation
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI | Async REST API server with automatic OpenAPI docs |
| **Runtime** | Python 3.11+ | Core language |
| **ASGI Server** | Uvicorn | Production-grade async server |
| **Speech-to-Text** | Sarvam AI STT API | Multilingual speech recognition (10+ Indian languages) |
| **Text-to-Speech** | Sarvam AI TTS API | Natural voice synthesis in Indian languages |
| **Translation** | Sarvam AI Translate API | Bidirectional translation between Indian languages and English |
| **Transliteration** | Sarvam AI Transliterate API | Native script to romanized Latin conversion |
| **Language Detection** | Sarvam AI + Script Analysis | Hybrid detection using Unicode ranges + API fallback |
| **Vector Database** | ChromaDB | Local vector store for RAG retrieval |
| **Embeddings** | Sentence Transformers (via ChromaDB) | Text embedding for semantic search |
| **LLM Brain** | Gemini (via Dev A) | Vendor response generation with personality |
| **Knowledge Graph** | Neo4j (via Dev A) | Structured vendor/product knowledge |
| **HTTP Client** | httpx | Async HTTP calls to Dev A and Sarvam APIs |
| **Data Validation** | Pydantic v2 | Request/response schema validation |
| **VR Frontend** | Unity + C# | Meta Quest 3 VR application |
| **Tunneling** | ngrok | Expose local servers for cross-machine communication |

---

## API Endpoints

### Health Check

```
GET /health
```

Returns service status.

```json
{
    "status": "ok",
    "service": "samvadxr-intelligence-engine"
}
```

### Main Pipeline (Unity Integration)

```
POST /api/test
```

Primary endpoint for Unity VR frontend. Accepts base64-encoded audio and returns vendor response with audio.

### Full Interact Pipeline

```
POST /api/interact
```

Full-featured pipeline endpoint with structured scene context and negotiation state tracking.

### Session Management

```
POST /api/reset
```

Resets the active session. Clears conversation history from RAG. Call when starting a new VR scene.

### RAG Management

```
POST /api/reload-rag
```

Force-reloads the RAG knowledge base from disk. Use after updating data files.

```
GET /api/debug/rag
```

Dumps all ChromaDB documents for inspection. Reports contaminated documents containing forbidden keywords.

---

## Request and Response Schemas

### /api/test

**Request (TestRequest)**

```json
{
    "audioData": "<base64-encoded mono WAV 16kHz>",
    "inputLanguage": "en",
    "targetLanguage": "ta",
    "object_grabbed": "tomato",
    "happiness_score": 50,
    "negotiation_state": "GREETING"
}
```

**Response (BackendFullResponse)**

```json
{
    "reply": "Vanakkam sagodharaa! Vaanga vaanga, innaiku ellaa kaaikari-pazhamum taazaa vandhirukkudhu!",
    "replyEnglish": "Hello brother! Come come, today all the vegetables and fruits have arrived fresh!",
    "audioReply": "<base64-encoded TTS audio in Tamil>",
    "negotiation_state": "INQUIRY",
    "happiness_score": 55,
    "suggested_user_response": "Annaachi, thakkali evvalavu?",
    "suggested_user_response_english": "Brother, how much are the tomatoes?"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `reply` | string | Vendor response transliterated to romanized Latin script |
| `replyEnglish` | string | English translation of the vendor response |
| `audioReply` | string | Base64-encoded TTS audio in the target language (native pronunciation) |
| `negotiation_state` | string | Updated negotiation stage |
| `happiness_score` | integer | Vendor happiness level (0-100) |
| `suggested_user_response` | string | Suggested next user response transliterated to romanized Latin |
| `suggested_user_response_english` | string | English translation of the suggested response |

### /api/interact

**Request (InteractRequest)**

```json
{
    "session_id": "vr-session-abc123",
    "audio_base64": "<base64-encoded audio>",
    "language_code": "ta-IN",
    "scene_context": {
        "looking_at": "tomato",
        "items_in_hand": ["tomato"],
        "vendor_happiness": 50,
        "vendor_patience": 70,
        "negotiation_stage": "INQUIRY"
    }
}
```

**Response (InteractResponse)**

```json
{
    "session_id": "vr-session-abc123",
    "transcribed_text": "Anna ithu evvalavu?",
    "agent_reply_text": "Idhu kilo naarpadhu roobaa, romba nalla thakkali!",
    "agent_reply_english": "This is forty rupees per kilo, very good tomatoes!",
    "agent_audio_base64": "<base64-encoded TTS audio in Tamil>",
    "vendor_mood": "friendly",
    "suggested_user_response": "Annaachi, konjam kammi pannunga!",
    "suggested_user_response_english": "Brother, reduce the price a bit!",
    "negotiation_state": {
        "item": "tomato",
        "quoted_price": 40,
        "vendor_happiness": 55,
        "vendor_patience": 70,
        "stage": "BARGAINING",
        "turn_count": 4,
        "deal_status": "negotiating"
    }
}
```

---

## Services

### voice_ops.py — Sarvam AI Integration

Handles all speech and language processing through Sarvam AI APIs:

| Function | API | Description |
|----------|-----|-------------|
| `transcribe_with_sarvam()` | STT | Converts audio bytes to text in specified language |
| `speak_with_sarvam()` | TTS | Converts native script text to audio bytes |
| `translate_with_sarvam()` | Translate | Translates text between language pairs |
| `transliterate_with_sarvam()` | Transliterate | Converts native script to romanized Latin |
| `detect_language_with_sarvam()` | Language ID | Identifies the language of input text |
| `detect_language_robust()` | Hybrid | Script-based detection with API fallback |
| `normalize_language_code()` | Utility | Normalizes language codes (e.g., `ta` to `ta-IN`) |
| `get_language_name()` | Utility | Returns human-readable language name |

### rag_ops.py — ChromaDB RAG Operations

Manages the vector knowledge base:

| Function | Description |
|----------|-------------|
| `initialize_knowledge_base()` | Loads and chunks seed data from `data/` directory into ChromaDB |
| `retrieve_context()` | Semantic search for relevant knowledge chunks |
| `add_conversation_to_rag()` | Indexes conversation turns for future retrieval |
| `clear_conversation_from_rag()` | Removes session-specific conversation chunks |

### context_memory.py — Conversation Memory

Sliding-window conversation memory per session:

| Method | Description |
|--------|-------------|
| `add_turn()` | Stores a user or vendor turn with metadata |
| `get_recent_turns()` | Retrieves last N turns |
| `get_context_block()` | Builds formatted context string for LLM prompt |

### middleware.py — Encoding Utilities

| Function | Description |
|----------|-------------|
| `base64_to_bytes()` | Decodes base64 string to raw bytes |
| `bytes_to_base64()` | Encodes raw bytes to base64 string |

### exceptions.py — Custom Exceptions

| Exception | Raised When |
|-----------|-------------|
| `SarvamServiceError` | Any Sarvam AI API call fails |
| `RAGServiceError` | ChromaDB operations fail |

---

## Session Management

Sessions are managed in-memory with automatic lifecycle handling:

- **New Session**: Created when `negotiation_state` is `""` (empty) or `"GREETING"`
- **Continue Session**: Any other `negotiation_state` value continues the active session
- **Reset Session**: `POST /api/reset` explicitly destroys the active session
- **Session ID Format**: `session-<8-char-hex-uuid>`

Each session maintains its own `ConversationMemory` instance with full turn history and metadata.

```
GREETING  -->  New session created (fresh memory)
INQUIRY   -->  Continue active session
BARGAINING --> Continue active session
COUNTER   -->  Continue active session
DEAL_CLOSED -> Continue active session
GREETING  -->  New session created (previous session discarded)
```

---

## RAG Pipeline

### Knowledge Base Initialization

On startup, the system:

1. Reads all `.txt` files from the `data/` directory
2. Chunks text into semantically meaningful segments
3. Embeds chunks using Sentence Transformers
4. Stores embeddings in ChromaDB (in-memory or persistent)

### Retrieval Flow

For each user interaction:

1. User's transcribed text (translated to English) is used as the query
2. ChromaDB performs semantic similarity search (top 5 results)
3. Retrieved chunks are concatenated into a `rag_context` string
4. Context is sent to Dev A alongside conversation history

### Conversation Indexing

After each vendor response:

1. The user-vendor turn pair is formatted as a document
2. Indexed into ChromaDB with session and turn metadata
3. Available for retrieval in future turns within the same or different sessions

---

## Language Processing Pipeline

For non-English target languages, the system performs a multi-stage language transformation:

```
User Audio (Tamil)
    |
    v
STT: "anna ithu evvalavu?" (Tamil text)
    |
    v
Translate to English: "Brother, how much is this?" (for Dev A)
    |
    v
Dev A responds in English/Hinglish: "Arre sahab, yeh 40 rupaye kilo hai!"
    |
    v
Translate to Tamil: "அரே சாஹிப், இது 40 ரூபாய் கிலோ!" (native script)
    |
    v
Transliterate to Latin: "Are saahib, idhu 40 roobai kilo!" (romanized)
    |
    v
Translate to English: "Hey sir, this is 40 rupees per kilo!" (caption)
    |
    v
TTS (uses native script): [Tamil audio output]
```

### Supported Languages

| Code | Language |
|------|----------|
| `hi-IN` | Hindi |
| `ta-IN` | Tamil |
| `te-IN` | Telugu |
| `kn-IN` | Kannada |
| `ml-IN` | Malayalam |
| `mr-IN` | Marathi |
| `bn-IN` | Bengali |
| `gu-IN` | Gujarati |
| `pa-IN` | Punjabi |
| `od-IN` | Odia |
| `en-IN` | English |

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Sarvam AI
SARVAM_API_KEY=your_sarvam_api_key_here

# Dev A Brain Endpoint
DEV_A_URL=https://your-ngrok-url.ngrok-free.app/api/dev/generate
DEV_A_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
```

| Variable | Default | Description |
|----------|---------|-------------|
| `SARVAM_API_KEY` | (required) | API key for Sarvam AI services |
| `DEV_A_URL` | ngrok URL | Dev A's brain endpoint URL |
| `DEV_A_TIMEOUT` | `30` | Timeout in seconds for Dev A HTTP calls |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Setup and Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Sarvam AI API key
- Dev A's brain service running (or use dummy fallback)

### Steps

1. Clone the repository:

```bash
git clone <repository-url>
cd SamvadXR_intelligence_engine
```

2. Create and activate virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Configure environment variables:

```powershell
cp .env.example .env
# Edit .env with your API keys and endpoints
```

5. Prepare knowledge base data:

Place vendor knowledge `.txt` files in the `data/` directory.

---

## Running the Application

### Development

```powershell
cd D:\Intelligence\SamvadXR_intelligence_engine
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Expose via ngrok

```powershell
ngrok http 8000
```

### Verify

```powershell
curl http://localhost:8000/health
```

### API Documentation

Once running, interactive API docs are available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Development Notes

### Fallback Mechanisms

The system implements graceful degradation at every stage:

| Component | Fallback Behavior |
|-----------|-------------------|
| Dev A brain | Falls back to hardcoded dummy responses with state transitions |
| RAG retrieval | Continues without context if ChromaDB fails |
| Translation | Returns original text if translation fails |
| Transliteration | Returns input text unchanged if transliteration fails |
| TTS | Returns empty audio string; frontend shows text-only |
| Language detection | Falls back to target language if detection fails |

### Logging

The application produces detailed structured logs at every pipeline step:

- Request/response summaries with timing data
- Dev A communication payload previews
- Language detection and translation traces
- RAG retrieval previews
- End-to-end latency breakdowns

### Vendor Domain Constraint

The system enforces a vegetable/fruit market domain constraint. A shop instruction is prepended to every Dev A request to prevent off-domain responses (no silk, brass, handicrafts).

---
