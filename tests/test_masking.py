"""Tests for engine/masking.py — Privacy-By-Design Masking.

Verifies that PII fields are tokenized and never leak into the masked
context, and that tokenization is deterministic.
"""

import pytest

from engine.masking import build_masked_context, tokenize_id


class TestTokenizeId:
    """Tests for the deterministic ID tokenization function."""

    def test_deterministic_same_input(self):
        """Same raw_id always produces the same token."""
        token1 = tokenize_id("ACCT_12345", prefix="ACCT")
        token2 = tokenize_id("ACCT_12345", prefix="ACCT")
        assert token1 == token2

    def test_different_inputs_different_tokens(self):
        """Different raw_ids produce different tokens."""
        t1 = tokenize_id("ACCT_11111")
        t2 = tokenize_id("ACCT_22222")
        assert t1 != t2

    def test_prefix_appears_in_token(self):
        """The prefix is included in the output token."""
        token = tokenize_id("some_id", prefix="TEST")
        assert token.startswith("TEST_")

    def test_token_format(self):
        """Token is prefix + underscore + 8 hex chars."""
        token = tokenize_id("x", prefix="P")
        parts = token.split("_", 1)
        assert len(parts) == 2
        assert parts[0] == "P"
        assert len(parts[1]) == 8
        assert all(c in "0123456789abcdef" for c in parts[1])

    def test_empty_string_input(self):
        """Empty string is a valid input — tokenization still works."""
        token = tokenize_id("")
        assert token.startswith("TXN_")
        assert len(token) == 12  # "TXN_" + 8 hex chars


class TestBuildMaskedContext:
    """Tests for the masked context builder."""

    def test_no_raw_pii_in_output(self):
        """Raw PII values (card_fingerprint, device_id, account_id) never
        appear in the masked context output."""
        txn = {
            "txn_id": "TXN_SECRET_123",
            "card_fingerprint": "CARD_SECRET_ABC",
            "device_id": "DEV_SECRET_XYZ",
            "account_id": "ACCT_SECRET_999",
            "amount": 1500.0,
            "timestamp": "2026-07-15T12:00:00",
            "account_created_at": "2025-01-01T00:00:00",
            "ip_region": "Mumbai",
            "billing_region": "Mumbai",
        }
        rule_breakdown = {"raw_risk_score": 45, "velocity": 10, "amount_pattern": 0, "account_age": 15, "geo_mismatch": 0}
        context = build_masked_context(txn, rule_breakdown)

        # Raw values must NOT appear anywhere in the context
        context_str = str(context)
        assert "CARD_SECRET_ABC" not in context_str
        assert "DEV_SECRET_XYZ" not in context_str
        assert "ACCT_SECRET_999" not in context_str
        assert "TXN_SECRET_123" not in context_str

    def test_tokens_replace_pii(self):
        """Tokenized versions of IDs appear in the context."""
        txn = {
            "txn_id": "TXN_001",
            "card_fingerprint": "CARD_001",
            "device_id": "DEV_001",
            "account_id": "ACCT_001",
            "amount": 100.0,
            "timestamp": "2026-07-15T12:00:00",
            "account_created_at": "2025-01-01T00:00:00",
        }
        rule_breakdown = {"raw_risk_score": 10}
        context = build_masked_context(txn, rule_breakdown)

        assert context["txn_token"] == tokenize_id("TXN_001", prefix="TXN")
        assert context["card_token"] == tokenize_id("CARD_001", prefix="CARD")
        assert context["account_token"] == tokenize_id("ACCT_001", prefix="ACCT")

    def test_amount_preserved(self):
        """Amount is passed through unmodified (not PII)."""
        txn = {
            "txn_id": "T1",
            "card_fingerprint": "C1",
            "device_id": "D1",
            "account_id": "A1",
            "amount": 42.50,
            "timestamp": "2026-07-15T12:00:00",
            "account_created_at": "2025-01-01T00:00:00",
        }
        context = build_masked_context(txn, {"raw_risk_score": 5})
        assert context["amount"] == 42.50

    def test_account_age_computed(self):
        """account_age_minutes is computed from timestamp - account_created_at."""
        txn = {
            "txn_id": "T1",
            "card_fingerprint": "C1",
            "device_id": "D1",
            "account_id": "A1",
            "amount": 100.0,
            "timestamp": "2026-07-15T12:30:00",
            "account_created_at": "2026-07-15T12:00:00",
        }
        context = build_masked_context(txn, {"raw_risk_score": 5})
        assert context["account_age_minutes"] == 30.0

    def test_graph_info_tokenized(self):
        """Connected account IDs in graph_info are tokenized, not raw."""
        txn = {
            "txn_id": "T1",
            "card_fingerprint": "C1",
            "device_id": "D1",
            "account_id": "A1",
            "amount": 100.0,
            "timestamp": "2026-07-15T12:00:00",
            "account_created_at": "2025-01-01T00:00:00",
        }
        graph_info = {
            "connected_accounts": [
                {"account_id": "ACCT_SECRET_FRIEND", "shared_signal": "device"},
            ]
        }
        context = build_masked_context(txn, {"raw_risk_score": 5}, graph_info)

        ring = context["ring_connections"]
        assert ring["ring_size"] == 2
        conn_token = ring["connected_account_tokens"][0]["account_token"]
        assert conn_token == tokenize_id("ACCT_SECRET_FRIEND", prefix="ACCT")
        assert "ACCT_SECRET_FRIEND" not in str(conn_token)

    def test_missing_timestamps_handled(self):
        """Missing timestamp or account_created_at results in None age."""
        txn = {
            "txn_id": "T1",
            "card_fingerprint": "C1",
            "device_id": "D1",
            "account_id": "A1",
            "amount": 100.0,
            "timestamp": None,
            "account_created_at": None,
        }
        context = build_masked_context(txn, {"raw_risk_score": 5})
        assert context["account_age_minutes"] is None

    def test_deterministic_output(self):
        """Same inputs produce identical masked context (deterministic)."""
        txn = {
            "txn_id": "TXN_DET",
            "card_fingerprint": "CARD_DET",
            "device_id": "DEV_DET",
            "account_id": "ACCT_DET",
            "amount": 999.99,
            "timestamp": "2026-07-15T12:00:00",
            "account_created_at": "2025-01-01T00:00:00",
        }
        rule_breakdown = {"raw_risk_score": 30}
        ctx1 = build_masked_context(txn, rule_breakdown)
        ctx2 = build_masked_context(txn, rule_breakdown)
        assert ctx1 == ctx2
