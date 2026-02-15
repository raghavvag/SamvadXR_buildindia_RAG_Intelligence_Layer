"""
Samvad XR — Pydantic Request/Response Models
=============================================
Dev B owns these schemas. Unity sends InteractRequest,
we return InteractResponse.
"""

from pydantic import BaseModel, Field


class SceneContext(BaseModel):
    """Spatial and game-state data sent by Unity each turn."""
    items_in_hand: list[str] = Field(default_factory=list)
    looking_at: str = ""
    distance_to_vendor: float = 0.0
    vendor_npc_id: str = "vendor_01"
    vendor_happiness: int = 50
    vendor_patience: int = 70
    negotiation_stage: str = "GREETING"
    current_price: int = 0
    user_offer: int = 0


class InteractRequest(BaseModel):
    """Incoming request from Unity VR client."""
    session_id: str
    audio_base64: str
    language_code: str = "hi-IN"
    scene_context: SceneContext = Field(default_factory=SceneContext)


class NegotiationState(BaseModel):
    """Game state snapshot returned to Unity."""
    item: str = ""
    quoted_price: int = 0
    vendor_happiness: int = 50
    vendor_patience: int = 70
    stage: str = "GREETING"
    turn_count: int = 0
    deal_status: str = "negotiating"


class InteractResponse(BaseModel):
    """Response sent back to Unity VR client."""
    session_id: str
    transcribed_text: str = ""
    agent_reply_text: str = ""
    agent_audio_base64: str = ""
    vendor_mood: str = "neutral"
    negotiation_state: NegotiationState = Field(default_factory=NegotiationState)
