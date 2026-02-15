"""
Tests for services/context_memory.py — Conversation Memory Layer.

Covers:
  - DialogueTurn dataclass creation
  - ConversationMemory construction (default & custom args)
  - add_turn (basic, metadata, timestamp)
  - get_context_block (formatting, text-only, turn numbering)
  - get_recent_turns (last N, edge cases)
  - Memory compression (_summarize_overflow, rolling summary)
  - clear (wipe all state)
  - Serialization roundtrip (to_dict / from_dict)
  - Edge cases (empty memory, single turn, large window)
"""

import time

import pytest

from services.context_memory import ConversationMemory, DialogueTurn


# ══════════════════════════════════════════════════════════════════════
#  DialogueTurn dataclass
# ══════════════════════════════════════════════════════════════════════

class TestDialogueTurn:
    """Tests for the DialogueTurn dataclass."""

    def test_create_with_all_fields(self):
        turn = DialogueTurn(role="user", text="Namaste", timestamp=1.0, metadata={"mood": 50})
        assert turn.role == "user"
        assert turn.text == "Namaste"
        assert turn.timestamp == 1.0
        assert turn.metadata == {"mood": 50}

    def test_defaults(self):
        turn = DialogueTurn(role="vendor", text="Hello")
        assert turn.timestamp == 0.0
        assert turn.metadata == {}

    def test_metadata_default_is_independent(self):
        """Each instance gets its own default dict (no shared mutable default)."""
        t1 = DialogueTurn(role="user", text="a")
        t2 = DialogueTurn(role="user", text="b")
        t1.metadata["key"] = "value"
        assert "key" not in t2.metadata


# ══════════════════════════════════════════════════════════════════════
#  Construction
# ══════════════════════════════════════════════════════════════════════

class TestConstruction:
    """Tests for ConversationMemory constructor."""

    def test_no_args_constructor(self):
        """No-args constructor works (execution plan requirement)."""
        mem = ConversationMemory()
        assert mem.session_id == ""
        assert mem.max_window == 10
        assert mem._turns == []
        assert mem._rolling_summary == ""

    def test_custom_session_id(self):
        mem = ConversationMemory(session_id="vr-session-abc123")
        assert mem.session_id == "vr-session-abc123"

    def test_custom_max_window(self):
        mem = ConversationMemory(max_window=5)
        assert mem.max_window == 5

    def test_full_constructor(self):
        mem = ConversationMemory(session_id="test-42", max_window=20)
        assert mem.session_id == "test-42"
        assert mem.max_window == 20


# ══════════════════════════════════════════════════════════════════════
#  add_turn
# ══════════════════════════════════════════════════════════════════════

class TestAddTurn:
    """Tests for add_turn method."""

    def test_adds_user_turn(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Namaste bhaiya!")
        assert len(mem._turns) == 1
        assert mem._turns[0].role == "user"
        assert mem._turns[0].text == "Namaste bhaiya!"

    def test_adds_vendor_turn(self):
        mem = ConversationMemory()
        mem.add_turn("vendor", "Aao aao! Kya chahiye?")
        assert mem._turns[0].role == "vendor"

    def test_metadata_stored_as_is(self):
        """Metadata is arbitrary dict, stored as-is (execution plan requirement)."""
        meta = {
            "held_item": "tomato",
            "looked_at_item": "mango",
            "vendor_happiness": 55,
            "vendor_patience": 70,
            "stage": "BROWSING",
        }
        mem = ConversationMemory()
        mem.add_turn("user", "Hello", metadata=meta)
        assert mem._turns[0].metadata == meta

    def test_metadata_none_defaults_to_empty_dict(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Hello")
        assert mem._turns[0].metadata == {}

    def test_timestamp_set_automatically(self):
        before = time.time()
        mem = ConversationMemory()
        mem.add_turn("user", "Hello")
        after = time.time()
        assert before <= mem._turns[0].timestamp <= after

    def test_multiple_turns_preserve_order(self):
        mem = ConversationMemory()
        mem.add_turn("user", "First")
        mem.add_turn("vendor", "Second")
        mem.add_turn("user", "Third")
        assert [t.text for t in mem._turns] == ["First", "Second", "Third"]


# ══════════════════════════════════════════════════════════════════════
#  get_context_block
# ══════════════════════════════════════════════════════════════════════

class TestGetContextBlock:
    """Tests for get_context_block — must return text-only, no metadata."""

    def test_empty_memory_returns_empty_string(self):
        mem = ConversationMemory()
        assert mem.get_context_block() == ""

    def test_three_turns_formatted_correctly(self):
        """Execution plan test: add 3 turns, verify all 3 appear formatted."""
        mem = ConversationMemory()
        mem.add_turn("user", "Namaste bhaiya!")
        mem.add_turn("vendor", "Aao aao! Kya chahiye?")
        mem.add_turn("user", "भाई ये tamatar कितने का है?")

        block = mem.get_context_block()
        assert "[Recent Dialogue]" in block
        assert "[Turn 1] User: Namaste bhaiya!" in block
        assert "[Turn 1] Vendor: Aao aao! Kya chahiye?" in block
        assert "[Turn 2] User: भाई ये tamatar कितने का है?" in block

    def test_text_only_no_metadata(self):
        """Context block must NOT contain metadata keys (execution plan: text-only)."""
        mem = ConversationMemory()
        mem.add_turn("user", "Hello", metadata={"mood": 55, "stage": "BROWSING"})
        block = mem.get_context_block()
        assert "mood" not in block
        assert "stage" not in block
        assert "BROWSING" not in block
        assert "55" not in block

    def test_single_user_turn(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Namaste!")
        block = mem.get_context_block()
        assert "[Turn 1] User: Namaste!" in block

    def test_includes_summary_when_present(self):
        mem = ConversationMemory()
        mem._rolling_summary = "The user greeted the vendor and asked about scarves."
        mem.add_turn("user", "Kitna hua?")
        block = mem.get_context_block()
        assert "[Summary of earlier conversation]" in block
        assert "The user greeted the vendor and asked about scarves." in block
        assert "[Recent Dialogue]" in block
        assert "Kitna hua?" in block

    def test_turn_numbering_groups_pairs(self):
        """User+Vendor in the same turn share the same turn number."""
        mem = ConversationMemory()
        mem.add_turn("user", "Hi")
        mem.add_turn("vendor", "Hello!")
        mem.add_turn("user", "Price?")
        mem.add_turn("vendor", "₹800")

        block = mem.get_context_block()
        lines = block.strip().split("\n")
        # Find turn-numbered lines
        numbered = [l for l in lines if l.startswith("[Turn")]
        assert numbered[0] == "[Turn 1] User: Hi"
        assert numbered[1] == "[Turn 1] Vendor: Hello!"
        assert numbered[2] == "[Turn 2] User: Price?"
        assert numbered[3] == "[Turn 2] Vendor: ₹800"

    def test_hindi_text_preserved(self):
        mem = ConversationMemory()
        mem.add_turn("user", "भाई ये कितने का है?")
        mem.add_turn("vendor", "₹40 lagega per kilo, bilkul taaza hai!")
        block = mem.get_context_block()
        assert "भाई ये कितने का है?" in block
        assert "₹40 lagega" in block


# ══════════════════════════════════════════════════════════════════════
#  get_recent_turns
# ══════════════════════════════════════════════════════════════════════

class TestGetRecentTurns:
    """Tests for get_recent_turns method."""

    def test_empty_memory(self):
        mem = ConversationMemory()
        assert mem.get_recent_turns() == []

    def test_returns_last_n_turns(self):
        mem = ConversationMemory()
        for i in range(8):
            mem.add_turn("user" if i % 2 == 0 else "vendor", f"Turn {i}")
        recent = mem.get_recent_turns(n=3)
        assert len(recent) == 3
        assert recent[0].text == "Turn 5"
        assert recent[1].text == "Turn 6"
        assert recent[2].text == "Turn 7"

    def test_n_larger_than_available(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Only one")
        recent = mem.get_recent_turns(n=10)
        assert len(recent) == 1

    def test_default_n_is_5(self):
        mem = ConversationMemory()
        for i in range(8):
            mem.add_turn("user", f"Turn {i}")
        recent = mem.get_recent_turns()
        assert len(recent) == 5

    def test_metadata_accessible_via_recent_turns(self):
        """Execution plan: metadata is accessible via get_recent_turns."""
        meta = {"held_item": "tomato", "stage": "HAGGLING"}
        mem = ConversationMemory()
        mem.add_turn("user", "₹40 final", metadata=meta)
        recent = mem.get_recent_turns(n=1)
        assert recent[0].metadata == meta
        assert recent[0].metadata["held_item"] == "tomato"


# ══════════════════════════════════════════════════════════════════════
#  Memory Compression (overflow / rolling summary)
# ══════════════════════════════════════════════════════════════════════

class TestMemoryCompression:
    """Tests for window overflow and rolling summary compression."""

    def test_no_compression_within_window(self):
        mem = ConversationMemory(max_window=10)
        for i in range(10):
            mem.add_turn("user" if i % 2 == 0 else "vendor", f"Turn {i}")
        assert len(mem._turns) == 10
        assert mem._rolling_summary == ""

    def test_compression_on_overflow(self):
        """Execution plan: add 12 turns (exceeds window=10), verify summary created."""
        mem = ConversationMemory(max_window=10)
        for i in range(12):
            role = "user" if i % 2 == 0 else "vendor"
            mem.add_turn(role, f"Turn {i}")

        # Should have compressed overflow turns into summary
        assert len(mem._turns) <= 10
        assert mem._rolling_summary != ""

    def test_compressed_turns_appear_in_summary(self):
        mem = ConversationMemory(max_window=5)
        mem.add_turn("user", "Hello!")
        mem.add_turn("vendor", "Welcome!")
        mem.add_turn("user", "Show me scarves")
        mem.add_turn("vendor", "Here you go")
        mem.add_turn("user", "How much?")
        # Window full (5 turns). Adding 6th triggers compression.
        mem.add_turn("vendor", "₹800")

        assert len(mem._turns) == 5
        assert "Hello!" in mem._rolling_summary

    def test_recent_turns_kept_after_compression(self):
        mem = ConversationMemory(max_window=5)
        for i in range(7):
            mem.add_turn("user", f"Turn {i}")
        # Most recent turns should still be in _turns
        texts = [t.text for t in mem._turns]
        assert "Turn 6" in texts  # last turn added
        assert "Turn 5" in texts

    def test_context_block_shows_summary_and_recent(self):
        """After compression, get_context_block includes both summary and recent dialogue."""
        mem = ConversationMemory(max_window=3)
        mem.add_turn("user", "Hello vendor!")
        mem.add_turn("vendor", "Welcome to my shop!")
        mem.add_turn("user", "Nice items here")
        # These trigger compressions
        mem.add_turn("vendor", "Thank you!")
        mem.add_turn("user", "How much for scarf?")

        block = mem.get_context_block()
        assert "[Summary of earlier conversation]" in block
        assert "[Recent Dialogue]" in block
        # Latest turns visible in recent dialogue
        assert "How much for scarf?" in block

    def test_multiple_compressions_accumulate(self):
        """Repeated overflows append to rolling summary."""
        mem = ConversationMemory(max_window=3)
        for i in range(9):
            mem.add_turn("user" if i % 2 == 0 else "vendor", f"Msg-{i}")

        assert mem._rolling_summary != ""
        assert len(mem._turns) <= 3
        # Earliest messages should be in summary
        assert "Msg-0" in mem._rolling_summary

    def test_small_window_stress(self):
        """Window of 2 with many turns — compression fires often."""
        mem = ConversationMemory(max_window=2)
        for i in range(20):
            mem.add_turn("user", f"Line {i}")
        assert len(mem._turns) <= 2
        assert mem._rolling_summary != ""
        # Most recent turns accessible
        recent = mem.get_recent_turns(n=2)
        assert recent[-1].text == "Line 19"


# ══════════════════════════════════════════════════════════════════════
#  clear
# ══════════════════════════════════════════════════════════════════════

class TestClear:
    """Tests for clear method."""

    def test_clear_resets_turns(self):
        """Execution plan: clear() resets everything."""
        mem = ConversationMemory()
        mem.add_turn("user", "Hello")
        mem.add_turn("vendor", "Hi")
        mem.clear()
        assert mem._turns == []

    def test_clear_resets_summary(self):
        mem = ConversationMemory(max_window=2)
        mem.add_turn("user", "A")
        mem.add_turn("vendor", "B")
        mem.add_turn("user", "C")  # triggers compression
        assert mem._rolling_summary != ""
        mem.clear()
        assert mem._rolling_summary == ""

    def test_clear_produces_empty_context_block(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Something")
        mem.clear()
        assert mem.get_context_block() == ""

    def test_clear_preserves_config(self):
        """Clearing wipes data but keeps session_id and max_window."""
        mem = ConversationMemory(session_id="keep-me", max_window=7)
        mem.add_turn("user", "X")
        mem.clear()
        assert mem.session_id == "keep-me"
        assert mem.max_window == 7


# ══════════════════════════════════════════════════════════════════════
#  Serialization: to_dict / from_dict
# ══════════════════════════════════════════════════════════════════════

class TestSerialization:
    """Tests for to_dict and from_dict roundtrip."""

    def test_roundtrip_preserves_all_data(self):
        """Execution plan: to_dict() → from_dict() roundtrip preserves all data."""
        mem = ConversationMemory(session_id="roundtrip-test", max_window=8)
        mem.add_turn("user", "Namaste!", metadata={"mood": 50})
        mem.add_turn("vendor", "Aao!", metadata={"stage": "GREETING"})

        d = mem.to_dict()
        restored = ConversationMemory.from_dict(d)

        assert restored.session_id == "roundtrip-test"
        assert restored.max_window == 8
        assert len(restored._turns) == 2
        assert restored._turns[0].text == "Namaste!"
        assert restored._turns[0].metadata == {"mood": 50}
        assert restored._turns[1].text == "Aao!"
        assert restored._turns[1].metadata == {"stage": "GREETING"}

    def test_roundtrip_preserves_rolling_summary(self):
        mem = ConversationMemory(max_window=3)
        for i in range(5):
            mem.add_turn("user", f"Turn {i}")
        assert mem._rolling_summary != ""

        d = mem.to_dict()
        restored = ConversationMemory.from_dict(d)
        assert restored._rolling_summary == mem._rolling_summary

    def test_to_dict_structure(self):
        mem = ConversationMemory(session_id="struct-test")
        mem.add_turn("user", "Hi")
        d = mem.to_dict()

        assert "session_id" in d
        assert "max_window" in d
        assert "rolling_summary" in d
        assert "turns" in d
        assert isinstance(d["turns"], list)
        assert d["turns"][0]["role"] == "user"
        assert d["turns"][0]["text"] == "Hi"
        assert "timestamp" in d["turns"][0]
        assert "metadata" in d["turns"][0]

    def test_from_dict_with_empty_data(self):
        mem = ConversationMemory.from_dict({})
        assert mem.session_id == ""
        assert mem.max_window == 10
        assert mem._turns == []
        assert mem._rolling_summary == ""

    def test_roundtrip_timestamps_preserved(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Time test")
        original_ts = mem._turns[0].timestamp

        restored = ConversationMemory.from_dict(mem.to_dict())
        assert restored._turns[0].timestamp == original_ts

    def test_context_block_identical_after_roundtrip(self):
        """Context blocks should be identical before and after serialization."""
        mem = ConversationMemory(max_window=3)
        for i in range(5):
            mem.add_turn("user" if i % 2 == 0 else "vendor", f"Message {i}")

        original_block = mem.get_context_block()
        restored = ConversationMemory.from_dict(mem.to_dict())
        restored_block = restored.get_context_block()

        assert original_block == restored_block


# ══════════════════════════════════════════════════════════════════════
#  Edge Cases
# ══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Misc edge cases and integration-style tests."""

    def test_empty_text_turn(self):
        """Empty string is valid (silence detection)."""
        mem = ConversationMemory()
        mem.add_turn("user", "")
        assert mem._turns[0].text == ""

    def test_unicode_and_special_chars(self):
        """Hindi script, rupee symbol, mixed Hinglish."""
        mem = ConversationMemory()
        mem.add_turn("user", "भाई ₹500 final है!")
        mem.add_turn("vendor", "Nahi nahi, ₹700 se neeche impossible! 😤")
        block = mem.get_context_block()
        assert "₹500" in block
        assert "₹700" in block
        assert "😤" in block

    def test_large_metadata(self):
        """Metadata can be arbitrarily complex."""
        meta = {
            "held_item": "tomato",
            "looked_at_items": ["onion", "potato"],
            "nested": {"a": {"b": {"c": 42}}},
            "tags": [1, 2, 3],
        }
        mem = ConversationMemory()
        mem.add_turn("user", "Complex", metadata=meta)
        assert mem._turns[0].metadata["nested"]["a"]["b"]["c"] == 42

    def test_concurrent_sessions_independent(self):
        """Two ConversationMemory instances don't share state."""
        m1 = ConversationMemory(session_id="s1")
        m2 = ConversationMemory(session_id="s2")
        m1.add_turn("user", "From session 1")
        m2.add_turn("user", "From session 2")
        assert len(m1._turns) == 1
        assert len(m2._turns) == 1
        assert m1._turns[0].text == "From session 1"
        assert m2._turns[0].text == "From session 2"

    def test_max_window_1(self):
        """Extreme case: window of 1."""
        mem = ConversationMemory(max_window=1)
        mem.add_turn("user", "First")
        mem.add_turn("user", "Second")
        assert len(mem._turns) == 1
        assert mem._turns[0].text == "Second"
        assert "First" in mem._rolling_summary

    def test_full_pipeline_simulation(self):
        """Simulate a realistic 6-turn bargaining session."""
        mem = ConversationMemory(session_id="bazaar-demo", max_window=10)

        exchanges = [
            ("user", "Namaste bhaiya!", {"stage": "GREETING"}),
            ("vendor", "Aao aao! Kya chahiye?", {"stage": "GREETING"}),
            ("user", "Ye tamatar dikhao", {"stage": "BROWSING", "looked_at_item": "tomato"}),
            ("vendor", "Bilkul taaza hai! ₹80 per kilo lagega", {"stage": "HAGGLING", "price": 80}),
            ("user", "Bahut mehnga! ₹30 do", {"stage": "HAGGLING"}),
            ("vendor", "₹30?! Itne mein toh mandi se bhi nahi ayega!", {"stage": "HAGGLING", "price": 60}),
            ("user", "Chalo ₹40 final", {"stage": "HAGGLING"}),
            ("vendor", "₹50 se neeche impossible", {"stage": "HAGGLING", "price": 50}),
            ("user", "Theek hai, ₹50 done", {"stage": "DEAL"}),
            ("vendor", "Bahut accha! Deal pakka!", {"stage": "DEAL", "price": 50}),
        ]

        for role, text, meta in exchanges:
            mem.add_turn(role, text, metadata=meta)

        block = mem.get_context_block()
        assert "[Recent Dialogue]" in block
        assert "₹50 done" in block
        assert "Deal pakka" in block

        # All 10 within window, no summary
        assert mem._rolling_summary == ""

        # Metadata preserved
        recent = mem.get_recent_turns(n=2)
        assert recent[-1].metadata.get("price") == 50

        # Serialization works
        d = mem.to_dict()
        restored = ConversationMemory.from_dict(d)
        assert restored.get_context_block() == block
