"""Shared fixtures for the Sentinel test suite."""

import json

import pandas as pd
import pytest


@pytest.fixture
def low_risk_transaction():
    """A clearly legitimate single transaction: old account, small amount,
    same region, no velocity."""
    return {
        "txn_id": "TXN_TEST_LOW",
        "card_fingerprint": "CARD_NORMAL",
        "device_id": "DEV_NORMAL",
        "ip_region": "Mumbai",
        "account_id": "ACCT_NORMAL",
        "account_created_at": "2025-01-01T00:00:00",
        "amount": 500.0,
        "timestamp": "2026-07-15T12:00:00",
        "billing_region": "Mumbai",
        "status": "approved",
        "decline_type": None,
        "true_label": False,
    }


@pytest.fixture
def high_risk_transaction():
    """A clearly fraudulent card-testing transaction: brand-new account,
    large amount, many same-card txns in short window, region mismatch."""
    return {
        "txn_id": "TXN_TEST_HIGH",
        "card_fingerprint": "CARD_FRAUD",
        "device_id": "DEV_FRAUD",
        "ip_region": "Delhi",
        "account_id": "ACCT_FRAUD",
        "account_created_at": "2026-07-15T11:55:00",
        "amount": 5000.0,
        "timestamp": "2026-07-15T12:00:00",
        "billing_region": "Chennai",
        "status": "declined",
        "decline_type": "stolen_card",
        "true_label": True,
    }


@pytest.fixture
def card_testing_sequence():
    """5 rapid same-card transactions simulating card testing.
    Returns a DataFrame with the txns plus the 'target' (last one).
    All prior txns are within 5 minutes of the target so velocity maxes out."""
    base_ts = pd.Timestamp("2026-07-15T12:00:00")
    txns = []
    for i in range(5):
        txns.append({
            "txn_id": f"TXN_VEL_{i}",
            "card_fingerprint": "CARD_VEL_TEST",
            "device_id": "DEV_VEL",
            "ip_region": "Mumbai",
            "account_id": f"ACCT_VEL_{i}",
            "account_created_at": "2025-06-01T00:00:00",
            "amount": 15.0 + i,  # small amounts (card testing pattern)
            # Spread within 4 minutes so all 5 are within 5min of target
            "timestamp": str(base_ts + pd.Timedelta(minutes=i * 0.8)),
            "billing_region": "Mumbai",
            "status": "approved",
            "decline_type": None,
            "true_label": True,
        })
    # Add a large final txn on the same card (the one we score)
    txns.append({
        "txn_id": "TXN_VEL_TARGET",
        "card_fingerprint": "CARD_VEL_TEST",
        "device_id": "DEV_VEL",
        "ip_region": "Mumbai",
        "account_id": "ACCT_VEL_TARGET",
        "account_created_at": "2025-06-01T00:00:00",
        "amount": 5000.0,
        "timestamp": str(base_ts + pd.Timedelta(minutes=5)),
        "billing_region": "Mumbai",
        "status": "declined",
        "decline_type": "stolen_card",
        "true_label": True,
    })
    df = pd.DataFrame(txns)
    df["account_created_at"] = pd.to_datetime(df["account_created_at"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def declined_transactions():
    """A small set of declined transactions covering soft and hard decline types."""
    return pd.DataFrame([
        {
            "txn_id": "TXN_SOFT_1",
            "card_fingerprint": "CARD_A",
            "device_id": "DEV_A",
            "ip_region": "Mumbai",
            "account_id": "ACCT_A",
            "account_created_at": "2025-01-01T00:00:00",
            "amount": 1500.0,
            "timestamp": "2026-07-15T12:00:00",
            "billing_region": "Mumbai",
            "status": "declined",
            "decline_type": "timeout",
            "true_label": False,
        },
        {
            "txn_id": "TXN_SOFT_2",
            "card_fingerprint": "CARD_B",
            "device_id": "DEV_B",
            "ip_region": "Delhi",
            "account_id": "ACCT_B",
            "account_created_at": "2025-01-01T00:00:00",
            "amount": 4000.0,
            "timestamp": "2026-07-15T12:05:00",
            "billing_region": "Delhi",
            "status": "declined",
            "decline_type": "insufficient_funds",
            "true_label": False,
        },
        {
            "txn_id": "TXN_HARD_1",
            "card_fingerprint": "CARD_C",
            "device_id": "DEV_C",
            "ip_region": "Chennai",
            "account_id": "ACCT_C",
            "account_created_at": "2025-01-01T00:00:00",
            "amount": 2500.0,
            "timestamp": "2026-07-15T12:10:00",
            "billing_region": "Chennai",
            "status": "declined",
            "decline_type": "stolen_card",
            "true_label": False,
        },
        {
            "txn_id": "TXN_HARD_2",
            "card_fingerprint": "CARD_D",
            "device_id": "DEV_D",
            "ip_region": "Bangalore",
            "account_id": "ACCT_D",
            "account_created_at": "2025-01-01T00:00:00",
            "amount": 8000.0,
            "timestamp": "2026-07-15T12:15:00",
            "billing_region": "Bangalore",
            "status": "declined",
            "decline_type": "expired_card",
            "true_label": False,
        },
    ])


@pytest.fixture
def sample_graph_df():
    """A small DataFrame with accounts that form a ring (A-B-C) plus
    an isolated account D. Used for graph_builder tests."""
    return pd.DataFrame([
        {
            "txn_id": "TXN_G1",
            "card_fingerprint": "CARD_G1",
            "device_id": "DEV_SHARED",
            "ip_region": "Mumbai",
            "account_id": "ACCT_RING_A",
            "account_created_at": "2025-01-01T00:00:00",
            "amount": 100.0,
            "timestamp": "2026-07-15T12:00:00",
            "billing_region": "Mumbai",
            "status": "approved",
            "decline_type": None,
            "true_label": False,
        },
        {
            "txn_id": "TXN_G2",
            "card_fingerprint": "CARD_G2",
            "device_id": "DEV_SHARED",
            "ip_region": "Mumbai",
            "account_id": "ACCT_RING_B",
            "account_created_at": "2025-01-01T00:00:00",
            "amount": 200.0,
            "timestamp": "2026-07-15T12:01:00",
            "billing_region": "Mumbai",
            "status": "approved",
            "decline_type": None,
            "true_label": False,
        },
        {
            "txn_id": "TXN_G3",
            "card_fingerprint": "CARD_G3",
            "device_id": "DEV_SHARED",
            "ip_region": "Mumbai",
            "account_id": "ACCT_RING_C",
            "account_created_at": "2025-01-01T00:00:00",
            "amount": 300.0,
            "timestamp": "2026-07-15T12:02:00",
            "billing_region": "Mumbai",
            "status": "approved",
            "decline_type": None,
            "true_label": False,
        },
        {
            "txn_id": "TXN_G4",
            "card_fingerprint": "CARD_G4",
            "device_id": "DEV_ISOLATED",
            "ip_region": "Delhi",
            "account_id": "ACCT_ISOLATED",
            "account_created_at": "2025-01-01T00:00:00",
            "amount": 400.0,
            "timestamp": "2026-07-15T12:03:00",
            "billing_region": "Delhi",
            "status": "approved",
            "decline_type": None,
            "true_label": False,
        },
    ])
