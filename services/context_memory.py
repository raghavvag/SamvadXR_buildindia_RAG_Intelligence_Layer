"""
Samvad XR — Conversation Memory
=================================
Instance-based conversation history manager.

One ConversationMemory per active VR session. Stores dialogue
turns with arbitrary metadata, provides text-only context blocks
for the LLM, and auto-compresses old turns into a rolling summary
when the window overflows.

All methods are sync (in-memory only). Dev B creates one per
session in main.py's session store.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("samvadxr.memory")


@dataclass
class DialogueTurn:
    """A single turn in the conversation."""
    role: str               # "user" | "vendor"
    text: str               # The spoken text (native script)
    timestamp: float = 0.0  # time.time() at creation
    metadata: dict = field(default_factory=dict)


class ConversationMemory:
    """Instance-based conversation history manager. One per session."""

    def __init__(self, session_id: str = "", max_window: int = 10):
        self.session_id = session_id
        self.max_window = max_window
        self._turns: list[DialogueTurn] = []
        self._rolling_summary: str = ""
        logger.debug(
            "ConversationMemory created: session=%s, max_window=%d",
            session_id or "(no-id)", max_window,
        )

    def add_turn(self, role: str, text: str, metadata: dict | None = None) -> None:
        """Append a dialogue turn. Triggers compression if window exceeded."""
        turn = DialogueTurn(
            role=role,
            text=text,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._turns.append(turn)
        logger.info(
            "Turn added: session=%s, role=%s, text=%.60s, turns_count=%d",
            self.session_id or "(no-id)", role, text, len(self._turns),
        )
        if len(self._turns) > self.max_window:
            logger.info(
                "Window overflow: %d turns > max %d — compressing",
                len(self._turns), self.max_window,
            )
            self._summarize_overflow()

    def get_context_block(self) -> str:
        """Return text-only conversation history (no metadata, no mood, no stage)."""
        parts = []
        if self._rolling_summary:
            parts.append(f"[Summary of earlier conversation]\n{self._rolling_summary}")
        if self._turns:
            parts.append("[Recent Dialogue]")
            turn_num = 1
            for i in range(0, len(self._turns), 2):
                # Group user+vendor pairs
                if i < len(self._turns):
                    parts.append(
                        f"[Turn {turn_num}] {self._turns[i].role.capitalize()}: "
                        f"{self._turns[i].text}"
                    )
                if i + 1 < len(self._turns):
                    parts.append(
                        f"[Turn {turn_num}] {self._turns[i + 1].role.capitalize()}: "
                        f"{self._turns[i + 1].text}"
                    )
                turn_num += 1
        block = "\n".join(parts)
        logger.debug(
            "Context block: session=%s, %d chars, %d recent turns, summary=%s",
            self.session_id or "(no-id)",
            len(block),
            len(self._turns),
            "yes" if self._rolling_summary else "no",
        )
        return block

    def get_recent_turns(self, n: int = 5) -> list[DialogueTurn]:
        """Return the last N turns."""
        return self._turns[-n:]

    def clear(self) -> None:
        """Wipe all turns and summary."""
        turn_count = len(self._turns)
        self._turns.clear()
        self._rolling_summary = ""
        logger.info(
            "Memory cleared: session=%s, removed %d turns",
            self.session_id or "(no-id)", turn_count,
        )

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "session_id": self.session_id,
            "max_window": self.max_window,
            "rolling_summary": self._rolling_summary,
            "turns": [
                {
                    "role": t.role,
                    "text": t.text,
                    "timestamp": t.timestamp,
                    "metadata": t.metadata,
                }
                for t in self._turns
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        """Deserialize from dict."""
        mem = cls(
            session_id=data.get("session_id", ""),
            max_window=data.get("max_window", 10),
        )
        mem._rolling_summary = data.get("rolling_summary", "")
        mem._turns = [
            DialogueTurn(**t) for t in data.get("turns", [])
        ]
        return mem

    def _summarize_overflow(self) -> None:
        """Compress oldest turns beyond window into rolling summary."""
        overflow_count = len(self._turns) - self.max_window
        if overflow_count <= 0:
            return
        # Take the overflow turns
        overflow = self._turns[:overflow_count]
        self._turns = self._turns[overflow_count:]
        # Append to rolling summary (hackathon: simple concat)
        summary_lines = [f"{t.role.capitalize()}: {t.text}" for t in overflow]
        if self._rolling_summary:
            self._rolling_summary += "\n" + "\n".join(summary_lines)
        else:
            self._rolling_summary = "\n".join(summary_lines)
        logger.info(
            "Compressed %d turns into summary (%d chars total), %d turns remain",
            overflow_count, len(self._rolling_summary), len(self._turns),
        )
