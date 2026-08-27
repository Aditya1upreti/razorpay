"""
DAY 4 (build FIRST, before ai_investigator.py) — Privacy-By-Design Masking

Nothing sensitive should ever be sent to the Claude API. This module
anonymizes/masks fields before they leave your pipeline.

Rules:
  - card_fingerprint / device_id / account_id -> already-hashed tokens,
    never raw values (do this at generation time too)
  - ip_region -> region-level only (e.g. "Delhi, IN"), never full IP
  - Only send: amounts, timestamps, rule scores, masked categorical
    signals, and tokenized IDs to the AI layer
"""

import hashlib
from datetime import datetime


def tokenize_id(raw_id: str, prefix: str = "TXN") -> str:
    """Deterministic short token for an ID, so the same raw_id always
    maps to the same token (needed to trace reasoning back to records)."""
    hash_hex = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
    return f"{prefix}_{hash_hex[:8]}"


def build_masked_context(transaction, rule_breakdown, graph_info=None) -> dict:
    """Assemble the exact payload that will be sent to the AI investigator.
    Must contain ONLY: tokenized IDs, amounts, timestamps, rule scores,
    and graph connection summary (also tokenized) — never raw PII."""
    ts = transaction.get("timestamp")
    created_at = transaction.get("account_created_at")

    account_age_minutes = None
    if ts and created_at:
        ts_dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
        ca_dt = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(str(created_at))
        account_age_minutes = round((ts_dt - ca_dt).total_seconds() / 60.0, 1)

    context = {
        "txn_token": tokenize_id(str(transaction.get("txn_id", "")), prefix="TXN"),
        "account_token": tokenize_id(str(transaction.get("account_id", "")), prefix="ACCT"),
        "card_token": tokenize_id(str(transaction.get("card_fingerprint", "")), prefix="CARD"),
        "amount": transaction.get("amount"),
        "timestamp": str(ts) if ts else None,
        "account_age_minutes": account_age_minutes,
        "rule_breakdown": rule_breakdown,
        "raw_risk_score": rule_breakdown.get("raw_risk_score", sum(rule_breakdown.get("rule_scores", {}).values())) if isinstance(rule_breakdown, dict) else None,
    }

    if isinstance(rule_breakdown, dict) and "raw_risk_score" not in context:
        context["raw_risk_score"] = rule_breakdown.get("raw_risk_score")

    if graph_info and isinstance(graph_info, dict):
        connected_accounts = graph_info.get("connected_accounts", [])
        masked_connections = []
        for conn in connected_accounts:
            masked_connections.append({
                "account_token": tokenize_id(str(conn.get("account_id", "")), prefix="ACCT"),
                "shared_signal": conn.get("shared_signal"),
            })
        context["ring_connections"] = {
            "connected_account_tokens": masked_connections,
            "ring_size": len(masked_connections) + 1,
        }

    return context
