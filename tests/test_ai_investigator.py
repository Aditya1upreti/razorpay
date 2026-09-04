"""Tests for engine/ai_investigator.py — Graceful Degradation (mocked).

Formalizes the existing ad-hoc degraded-mode test from ai_investigator.py's
__main__ block into a proper pytest test. Uses monkeypatch to simulate API
failure — no live Gemini calls are made.

This test validates that the EXACT same fallback path used in production
is triggered when the API call fails.
"""

import json
import pytest

from engine import ai_investigator


class TestGracefulDegradation:
    """Verify that investigate() falls back to rule-only scoring
    when the Gemini API call raises an exception."""

    def test_degraded_mode_on_api_failure(self, monkeypatch):
        """Forcing _call_gemini to raise RuntimeError triggers the fallback
        path, returning degraded_mode=True with rule-based verdict."""
        # The exact same mock approach as the ad-hoc test in ai_investigator.py
        def _broken_call_gemini(*args, **kwargs):
            raise RuntimeError("Simulated API failure for testing")

        monkeypatch.setattr(ai_investigator, "_call_gemini", _broken_call_gemini)

        fake_txn = {
            "txn_id": "TXN_DEGRADED_TEST_PYTEST",
            "account_id": "ACCT_FAKE",
            "amount": 100,
            "timestamp": "2026-01-01T00:00:00",
            "account_created_at": "2025-01-01T00:00:00",
            "card_fingerprint": "CARD_FAKE",
        }

        result = ai_investigator.investigate(
            fake_txn,
            {"rule_scores": {"velocity": 20}, "raw_risk_score": 50},
            force_refresh=True,
        )

        assert result["degraded_mode"] is True
        assert result["verdict"] in ("fraud_likely", "insufficient_evidence")
        assert result["recommended_action"] in ("block", "manual_review")
        assert result["confidence"] is None
        assert "AI investigator unavailable" in result["reasoning"]

    def test_degraded_mode_high_score_blocks(self, monkeypatch):
        """When raw_risk_score > 75, the fallback verdict is fraud_likely/block."""
        def _broken_call_gemini(*args, **kwargs):
            raise RuntimeError("Simulated API failure")

        monkeypatch.setattr(ai_investigator, "_call_gemini", _broken_call_gemini)

        fake_txn = {
            "txn_id": "TXN_DEGRADED_HIGH",
            "account_id": "ACCT_HIGH",
            "amount": 5000,
            "timestamp": "2026-01-01T00:00:00",
            "account_created_at": "2025-01-01T00:00:00",
            "card_fingerprint": "CARD_HIGH",
        }

        result = ai_investigator.investigate(
            fake_txn,
            {"raw_risk_score": 80},
            force_refresh=True,
        )

        assert result["degraded_mode"] is True
        assert result["verdict"] == "fraud_likely"
        assert result["recommended_action"] == "block"

    def test_degraded_mode_low_score_reviews(self, monkeypatch):
        """When raw_risk_score <= 75, the fallback verdict is insufficient_evidence/manual_review."""
        def _broken_call_gemini(*args, **kwargs):
            raise RuntimeError("Simulated API failure")

        monkeypatch.setattr(ai_investigator, "_call_gemini", _broken_call_gemini)

        fake_txn = {
            "txn_id": "TXN_DEGRADED_LOW",
            "account_id": "ACCT_LOW",
            "amount": 200,
            "timestamp": "2026-01-01T00:00:00",
            "account_created_at": "2025-01-01T00:00:00",
            "card_fingerprint": "CARD_LOW",
        }

        result = ai_investigator.investigate(
            fake_txn,
            {"raw_risk_score": 40},
            force_refresh=True,
        )

        assert result["degraded_mode"] is True
        assert result["verdict"] == "insufficient_evidence"
        assert result["recommended_action"] == "manual_review"
