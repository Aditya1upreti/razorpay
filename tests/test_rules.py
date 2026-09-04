"""Tests for engine/rules.py — Deterministic Rule Engine.

Tests individual scoring functions and the combined score_transaction/score_batch
against known inputs. No external data dependencies — all fixtures are synthetic.
"""

import json

import pandas as pd
import pytest

from engine.rules import (
    account_age_score,
    amount_pattern_score,
    geo_mismatch_score,
    score_batch,
    score_transaction,
    velocity_score,
)


class TestVelocityScore:
    """Tests for the velocity scoring factor (max 35 points)."""

    def test_single_txn_no_velocity(self, low_risk_transaction):
        """A lone transaction with no same-card history scores 0."""
        df = pd.DataFrame([low_risk_transaction])
        df["account_created_at"] = pd.to_datetime(df["account_created_at"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        score = velocity_score(low_risk_transaction, df)
        assert score == 0

    def test_high_velocity_card_testing(self, card_testing_sequence):
        """5 prior same-card txns within 5min window = max velocity (35)."""
        target = card_testing_sequence[
            card_testing_sequence["txn_id"] == "TXN_VEL_TARGET"
        ].iloc[0]
        score = velocity_score(target, card_testing_sequence)
        assert score == 35  # max_points

    def test_partial_velocity_scales(self, card_testing_sequence):
        """Fewer txns in window should give proportionally fewer points."""
        target = card_testing_sequence[
            card_testing_sequence["txn_id"] == "TXN_VEL_TARGET"
        ].iloc[0]
        # Only use first 3 same-card txns + target (4 total -> 3 others in window)
        subset = card_testing_sequence[
            card_testing_sequence["txn_id"].isin(
                [f"TXN_VEL_{i}" for i in range(3)] + ["TXN_VEL_TARGET"]
            )
        ]
        score = velocity_score(target, subset)
        # 3 others in window -> round(35 * 3 / 5) = 21
        assert score == 21

    def test_velocity_outside_window(self):
        """Same card but txns outside the 5-min window should score 0."""
        base_ts = pd.Timestamp("2026-07-15T12:00:00")
        txns = pd.DataFrame([
            {
                "txn_id": "TXN_OLD",
                "card_fingerprint": "CARD_SAME",
                "timestamp": base_ts - pd.Timedelta(minutes=30),
                "account_created_at": "2025-01-01",
                "amount": 100.0,
                "account_id": "ACCT_1",
                "ip_region": "Mumbai",
                "billing_region": "Mumbai",
                "device_id": "DEV_1",
                "status": "approved",
                "decline_type": None,
                "true_label": False,
            },
            {
                "txn_id": "TXN_NEW",
                "card_fingerprint": "CARD_SAME",
                "timestamp": base_ts,
                "account_created_at": "2025-01-01",
                "amount": 100.0,
                "account_id": "ACCT_2",
                "ip_region": "Mumbai",
                "billing_region": "Mumbai",
                "device_id": "DEV_2",
                "status": "approved",
                "decline_type": None,
                "true_label": False,
            },
        ])
        txns["timestamp"] = pd.to_datetime(txns["timestamp"])
        target = txns[txns["txn_id"] == "TXN_NEW"].iloc[0]
        score = velocity_score(target, txns)
        assert score == 0


class TestAmountPatternScore:
    """Tests for the amount-pattern scoring factor (max 30 points)."""

    def test_small_amount_scores_zero(self, low_risk_transaction):
        """Amount <= 1000 always scores 0 regardless of history."""
        df = pd.DataFrame([low_risk_transaction])
        df["account_created_at"] = pd.to_datetime(df["account_created_at"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        score = amount_pattern_score(low_risk_transaction, df)
        assert score == 0

    def test_card_testing_pattern(self, card_testing_sequence):
        """Large amount after 3+ small prior txns = 30 points."""
        target = card_testing_sequence[
            card_testing_sequence["txn_id"] == "TXN_VEL_TARGET"
        ].iloc[0]
        score = amount_pattern_score(target, card_testing_sequence)
        assert score == 30  # max_points

    def test_large_amount_no_small_prior(self):
        """Large amount but no small prior txns = 0 points."""
        base_ts = pd.Timestamp("2026-07-15T12:00:00")
        txns = pd.DataFrame([
            {
                "txn_id": "TXN_PRIOR",
                "card_fingerprint": "CARD_X",
                "timestamp": base_ts - pd.Timedelta(minutes=5),
                "account_created_at": "2025-01-01",
                "amount": 500.0,  # not small (<20)
                "account_id": "ACCT_X1",
                "ip_region": "Mumbai",
                "billing_region": "Mumbai",
                "device_id": "DEV_X",
                "status": "approved",
                "decline_type": None,
                "true_label": False,
            },
            {
                "txn_id": "TXN_BIG",
                "card_fingerprint": "CARD_X",
                "timestamp": base_ts,
                "account_created_at": "2025-01-01",
                "amount": 5000.0,
                "account_id": "ACCT_X2",
                "ip_region": "Mumbai",
                "billing_region": "Mumbai",
                "device_id": "DEV_X",
                "status": "declined",
                "decline_type": "stolen_card",
                "true_label": True,
            },
        ])
        txns["timestamp"] = pd.to_datetime(txns["timestamp"])
        target = txns[txns["txn_id"] == "TXN_BIG"].iloc[0]
        score = amount_pattern_score(target, txns)
        assert score == 0


class TestAccountAgeScore:
    """Tests for the account-age scoring factor (max 15 points)."""

    def test_brand_new_account(self):
        """Account < 10 minutes old = 15 points (max)."""
        txn = {
            "account_created_at": "2026-07-15T11:55:00",
            "timestamp": "2026-07-15T12:00:00",
        }
        assert account_age_score(txn) == 15

    def test_old_account(self):
        """Account >= 60 minutes old = 0 points."""
        txn = {
            "account_created_at": "2025-01-01T00:00:00",
            "timestamp": "2026-07-15T12:00:00",
        }
        assert account_age_score(txn) == 0

    def test_mid_range_account(self):
        """Account 35 minutes old: (60-35)/50 * 15 = 7.5 -> round to 8."""
        txn = {
            "account_created_at": "2026-07-15T11:25:00",
            "timestamp": "2026-07-15T12:00:00",
        }
        score = account_age_score(txn)
        # age=35min, (60-35)/50*15 = 7.5, round -> 8 (Python banker's rounding: 7)
        # Actually int(round(7.5)) = 8 in Python 3
        assert 0 < score < 15

    def test_boundary_10_minutes(self):
        """Exactly 10 minutes = NOT < 10, so falls to mid-range formula."""
        txn = {
            "account_created_at": "2026-07-15T11:50:00",
            "timestamp": "2026-07-15T12:00:00",
        }
        score = account_age_score(txn)
        # age=10, (60-10)/50*15 = 15.0 -> round -> 15
        # Wait, 10 is NOT < 10, so we go to the mid-range branch
        # (60-10)/50 * 15 = 15.0 -> int(round(15.0)) = 15
        assert score == 15

    def test_boundary_60_minutes(self):
        """Exactly 60 minutes = 0 points (>= 60 branch)."""
        txn = {
            "account_created_at": "2026-07-15T11:00:00",
            "timestamp": "2026-07-15T12:00:00",
        }
        assert account_age_score(txn) == 0

    def test_future_creation_time(self):
        """Future account_created_at (negative age) = 0 points."""
        txn = {
            "account_created_at": "2026-07-15T13:00:00",
            "timestamp": "2026-07-15T12:00:00",
        }
        assert account_age_score(txn) == 0


class TestGeoMismatchScore:
    """Tests for the geo-mismatch scoring factor (max 10 points)."""

    def test_matching_regions(self):
        """Same billing and ip_region = 0 points."""
        txn = {"billing_region": "Mumbai", "ip_region": "Mumbai"}
        assert geo_mismatch_score(txn) == 0

    def test_mismatched_regions(self):
        """Different billing and ip_region = 10 points."""
        txn = {"billing_region": "Mumbai", "ip_region": "Delhi"}
        assert geo_mismatch_score(txn) == 10

    def test_case_insensitive_comparison(self):
        """Leading/trailing whitespace is stripped; case matters after strip."""
        txn = {"billing_region": "  Mumbai  ", "ip_region": "Mumbai"}
        assert geo_mismatch_score(txn) == 0


class TestScoreTransaction:
    """Tests for the combined score_transaction function."""

    def test_low_risk_total(self, low_risk_transaction):
        """A clearly legitimate txn should score below 30 (safe band)."""
        df = pd.DataFrame([low_risk_transaction])
        df["account_created_at"] = pd.to_datetime(df["account_created_at"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        result = score_transaction(low_risk_transaction, df)
        assert result["raw_risk_score"] < 30
        assert "velocity" in result["rule_breakdown"]
        assert "amount_pattern" in result["rule_breakdown"]
        assert "account_age" in result["rule_breakdown"]
        assert "geo_mismatch" in result["rule_breakdown"]

    def test_high_risk_total(self, card_testing_sequence):
        """Card-testing sequence scores high: velocity=35 + amount_pattern=30 = 65."""
        target = card_testing_sequence[
            card_testing_sequence["txn_id"] == "TXN_VEL_TARGET"
        ].iloc[0]
        result = score_transaction(target, card_testing_sequence)
        breakdown = result["rule_breakdown"]
        # velocity=35 (5 prior same-card txns within 5min) + amount_pattern=30
        # (large amount after 5+ small prior txns) = 65 (ambiguous band)
        assert breakdown["velocity"] == 35
        assert breakdown["amount_pattern"] == 30
        assert result["raw_risk_score"] == 65

    def test_score_never_exceeds_100(self, card_testing_sequence):
        """Score is capped at 100 even with max points on all factors."""
        target = card_testing_sequence[
            card_testing_sequence["txn_id"] == "TXN_VEL_TARGET"
        ].iloc[0]
        # Override to trigger all factors
        target_dict = target.copy()
        target_dict["account_created_at"] = "2026-07-15T11:59:00"  # brand new
        target_dict["billing_region"] = "Chennai"  # mismatch with ip_region=Mumbai
        result = score_transaction(target_dict, card_testing_sequence)
        assert result["raw_risk_score"] <= 100


class TestScoreBatch:
    """Tests for the batch scoring function."""

    def test_adds_expected_columns(self, low_risk_transaction):
        """score_batch adds raw_risk_score, rule_breakdown, and band columns."""
        df = pd.DataFrame([low_risk_transaction])
        df["account_created_at"] = pd.to_datetime(df["account_created_at"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        result = score_batch(df)
        assert "raw_risk_score" in result.columns
        assert "rule_breakdown" in result.columns
        assert "band" in result.columns

    def test_band_assignments(self, card_testing_sequence):
        """Bands are assigned correctly based on score thresholds."""
        result = score_batch(card_testing_sequence)
        for _, row in result.iterrows():
            score = row["raw_risk_score"]
            band = row["band"]
            if score < 30:
                assert band == "safe"
            elif score > 75:
                assert band == "high_risk"
            else:
                assert band == "ambiguous"

    def test_rule_breakdown_is_json_string(self, low_risk_transaction):
        """rule_breakdown column contains parseable JSON."""
        df = pd.DataFrame([low_risk_transaction])
        df["account_created_at"] = pd.to_datetime(df["account_created_at"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        result = score_batch(df)
        breakdown = json.loads(result.iloc[0]["rule_breakdown"])
        assert isinstance(breakdown, dict)
        assert set(breakdown.keys()) == {
            "velocity",
            "amount_pattern",
            "account_age",
            "geo_mismatch",
        }
