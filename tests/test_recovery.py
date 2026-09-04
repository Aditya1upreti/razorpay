"""Tests for engine/recovery.py — Revenue Recovery.

Tests routing logic, retry/outreach simulation, suppression-list awareness,
and batch recovery behavior. No external data dependencies.
"""

import pandas as pd
import pytest

from engine.recovery import (
    HARD_DECLINE_TYPES,
    SOFT_DECLINE_TYPES,
    route_decline,
    run_recovery_batch,
    simulate_customer_outreach,
    simulate_silent_retry,
)


class TestRouteDecline:
    """Tests for decline-type routing logic."""

    @pytest.mark.parametrize("decline_type", SOFT_DECLINE_TYPES)
    def test_soft_declines_route_to_silent_retry(self, decline_type):
        """Soft decline types (timeout, insufficient_funds) -> silent_retry."""
        txn = {"decline_type": decline_type}
        assert route_decline(txn) == "silent_retry"

    @pytest.mark.parametrize("decline_type", HARD_DECLINE_TYPES)
    def test_hard_declines_route_to_outreach(self, decline_type):
        """Hard decline types (stolen_card, expired_card) -> customer_outreach."""
        txn = {"decline_type": decline_type}
        assert route_decline(txn) == "customer_outreach"

    def test_unknown_decline_type_raises(self):
        """An unexpected decline_type raises ValueError."""
        txn = {"decline_type": "magnetic_strip_fail"}
        with pytest.raises(ValueError, match="Unexpected decline_type"):
            route_decline(txn)


class TestSimulateSilentRetry:
    """Tests for silent retry simulation (soft declines)."""

    def test_small_amount_recovers_immediately(self):
        """Amount < 2000 succeeds on attempt 1."""
        txn = {"txn_id": "TXN_S1", "amount": 1500.0}
        result = simulate_silent_retry(txn)
        assert result["recovered"] is True
        assert result["attempts_made"] == 1
        assert result["recovered_amount"] == 1500.0

    def test_mid_range_amount_recovers_on_second_attempt(self):
        """Amount 2000-6000 needs retry, succeeds on attempt 2."""
        txn = {"txn_id": "TXN_S2", "amount": 4000.0}
        result = simulate_silent_retry(txn)
        assert result["recovered"] is True
        assert result["attempts_made"] == 2
        assert result["recovered_amount"] == 4000.0

    def test_large_amount_fails_all_attempts(self):
        """Amount > 6000 fails all attempts."""
        txn = {"txn_id": "TXN_S3", "amount": 8000.0}
        result = simulate_silent_retry(txn)
        assert result["recovered"] is False
        assert result["attempts_made"] == 2  # MAX_SILENT_RETRIES
        assert result["recovered_amount"] == 0.0

    def test_respects_custom_max_attempts(self):
        """Custom max_attempts caps the retry loop."""
        txn = {"txn_id": "TXN_S4", "amount": 4000.0}
        result = simulate_silent_retry(txn, max_attempts=1)
        assert result["attempts_made"] == 1
        assert result["recovered"] is False  # mid-range needs 2 attempts

    def test_returns_txn_id(self):
        """Result includes the original txn_id."""
        txn = {"txn_id": "TXN_ABC", "amount": 500.0}
        result = simulate_silent_retry(txn)
        assert result["txn_id"] == "TXN_ABC"


class TestSimulateCustomerOutreach:
    """Tests for customer outreach simulation (hard declines)."""

    def test_small_amount_recovers_immediately(self):
        """Amount < 3000 succeeds on attempt 1."""
        txn = {"txn_id": "TXN_H1", "account_id": "ACCT_1", "amount": 2500.0}
        result = simulate_customer_outreach(txn)
        assert result["recovered"] is True
        assert result["attempts_made"] == 1
        assert result["suppressed"] is False

    def test_mid_range_recovers_on_second_attempt(self):
        """Amount 3000-4500 succeeds on attempt 2."""
        txn = {"txn_id": "TXN_H2", "account_id": "ACCT_2", "amount": 3500.0}
        result = simulate_customer_outreach(txn)
        assert result["recovered"] is True
        assert result["attempts_made"] == 2

    def test_large_amount_fails(self):
        """Amount > 4500 fails all attempts."""
        txn = {"txn_id": "TXN_H3", "account_id": "ACCT_3", "amount": 8000.0}
        result = simulate_customer_outreach(txn)
        assert result["recovered"] is False
        assert result["attempts_made"] == 2  # MAX_OUTREACH_ATTEMPTS

    def test_suppressed_account_skips_outreach(self):
        """Account in suppression_list gets suppressed=True, no attempts."""
        txn = {"txn_id": "TXN_SUP", "account_id": "ACCT_SUPPRESSED", "amount": 2000.0}
        suppression = {"ACCT_SUPPRESSED"}
        result = simulate_customer_outreach(txn, suppression_list=suppression)
        assert result["suppressed"] is True
        assert result["attempts_made"] == 0
        assert result["recovered"] is False
        assert result["recovered_amount"] == 0.0

    def test_suppression_list_none_not_checked(self):
        """When suppression_list is None, no suppression check occurs."""
        txn = {"txn_id": "TXN_NS", "account_id": "ACCT_ANY", "amount": 2000.0}
        result = simulate_customer_outreach(txn, suppression_list=None)
        assert result["suppressed"] is False

    def test_respects_custom_max_attempts(self):
        """Custom max_attempts caps the outreach loop."""
        txn = {"txn_id": "TXN_H4", "account_id": "ACCT_4", "amount": 3500.0}
        result = simulate_customer_outreach(txn, max_attempts=1)
        assert result["attempts_made"] == 1
        assert result["recovered"] is False  # mid-range needs 2 attempts


class TestRunRecoveryBatch:
    """Tests for batch recovery processing."""

    def test_counts_by_route(self, declined_transactions):
        """Soft and hard decline counts are tracked separately."""
        summary = run_recovery_batch(declined_transactions)
        assert summary["soft_decline_count"] == 2  # timeout, insufficient_funds
        assert summary["hard_decline_count"] == 2  # stolen_card, expired_card

    def test_total_at_risk(self, declined_transactions):
        """total_at_risk sums all declined transaction amounts."""
        summary = run_recovery_batch(declined_transactions)
        expected = 1500.0 + 4000.0 + 2500.0 + 8000.0
        assert summary["total_at_risk"] == pytest.approx(expected)

    def test_suppression_reduces_hard_decline_recovery(self, declined_transactions):
        """Suppressing a hard-decline account reduces hard_decline_recovered."""
        # Suppress ACCT_C (stolen_card, amount=2500, would recover)
        suppressed = run_recovery_batch(
            declined_transactions, suppression_list={"ACCT_C"}
        )
        unsuppressed = run_recovery_batch(declined_transactions, suppression_list=None)

        assert suppressed["suppressed_count"] == 1
        assert suppressed["hard_decline_recovered"] < unsuppressed["hard_decline_recovered"]

    def test_per_transaction_log_length(self, declined_transactions):
        """Per-transaction log has one entry per declined transaction."""
        summary = run_recovery_batch(declined_transactions)
        assert len(summary["per_transaction_log"]) == len(declined_transactions)

    def test_per_transaction_log_has_required_fields(self, declined_transactions):
        """Each log entry has txn_id, route, attempts_made, recovered."""
        summary = run_recovery_batch(declined_transactions)
        for entry in summary["per_transaction_log"]:
            assert "txn_id" in entry
            assert "route" in entry
            assert "attempts_made" in entry
            assert "recovered" in entry
            assert entry["route"] in ("silent_retry", "customer_outreach")

    def test_empty_batch(self):
        """Empty DataFrame produces zero totals and empty log."""
        empty_df = pd.DataFrame(
            columns=[
                "txn_id", "card_fingerprint", "device_id", "ip_region",
                "account_id", "account_created_at", "amount", "timestamp",
                "billing_region", "status", "decline_type", "true_label",
            ]
        )
        summary = run_recovery_batch(empty_df)
        assert summary["total_at_risk"] == 0.0
        assert summary["total_recovered"] == 0.0
        assert summary["per_transaction_log"] == []

    def test_soft_decline_recovered_tracks_amount(self, declined_transactions):
        """soft_decline_recovered sums amounts of successfully recovered soft declines."""
        summary = run_recovery_batch(declined_transactions)
        # TXN_SOFT_1: amount=1500 (<2000, recovers on attempt 1) -> 1500
        # TXN_SOFT_2: amount=4000 (2000-6000, recovers on attempt 2) -> 4000
        assert summary["soft_decline_recovered"] == pytest.approx(5500.0)
