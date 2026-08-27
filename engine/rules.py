"""
DAY 2 — Layer 1: Deterministic Rule Engine

Fast, explainable, no AI. Every transaction gets a raw_risk_score (0-100)
built from independently-scored rules. Score bands:

  score < 30        -> auto-clear (safe)
  score > 75         -> auto-flag (high risk)
  30 <= score <= 75  -> ambiguous, send to Layer 2/3

Tune thresholds using ONLY the train set. Never touch the test set here.
"""

import json
from datetime import timedelta

import pandas as pd


def _parse_dt(val):
    if isinstance(val, pd.Timestamp):
        return val
    return pd.Timestamp(val)


def velocity_score(transaction, all_transactions, window_minutes=5, max_points=35):
    card = transaction["card_fingerprint"]
    ts = _parse_dt(transaction["timestamp"])
    window = timedelta(minutes=window_minutes)

    same_card = all_transactions[all_transactions["card_fingerprint"] == card]
    diffs = same_card["timestamp"].apply(lambda t: abs(_parse_dt(t) - ts))
    count = (diffs <= window).sum() - 1  # exclude self

    if count <= 0:
        return 0
    if count >= 5:
        return max_points
    return int(round(max_points * count / 5))


def amount_pattern_score(transaction, all_transactions, max_points=30):
    card = transaction["card_fingerprint"]
    ts = _parse_dt(transaction["timestamp"])
    amount = transaction["amount"]

    if amount <= 1000:
        return 0

    same_card = all_transactions[all_transactions["card_fingerprint"] == card].copy()
    same_card["_ts"] = same_card["timestamp"].apply(_parse_dt)
    same_card = same_card.sort_values("_ts")

    window_start = ts - timedelta(minutes=10)
    prior = same_card[(same_card["_ts"] >= window_start) & (same_card["_ts"] < ts)]

    small_count = (prior["amount"] < 20).sum()

    if small_count >= 3:
        return max_points
    return 0


def account_age_score(transaction, max_points=15):
    created = _parse_dt(transaction["account_created_at"])
    ts = _parse_dt(transaction["timestamp"])
    age_minutes = (ts - created).total_seconds() / 60.0

    if age_minutes < 0:
        return 0
    if age_minutes < 10:
        return max_points
    if age_minutes >= 60:
        return 0
    return int(round(max_points * (60 - age_minutes) / 50))


def geo_mismatch_score(transaction, max_points=10):
    billing = str(transaction["billing_region"]).strip()
    ip = str(transaction["ip_region"]).strip()
    if billing != ip:
        return max_points
    return 0


def score_transaction(transaction, all_transactions):
    """Returns dict: {raw_risk_score, rule_breakdown: {rule_name: points}}"""
    v = velocity_score(transaction, all_transactions)
    a = amount_pattern_score(transaction, all_transactions)
    ac = account_age_score(transaction)
    g = geo_mismatch_score(transaction)

    total = min(v + a + ac + g, 100)

    return {
        "raw_risk_score": total,
        "rule_breakdown": {
            "velocity": v,
            "amount_pattern": a,
            "account_age": ac,
            "geo_mismatch": g,
        },
    }


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Apply score_transaction to every row, add columns:
    raw_risk_score, rule_breakdown (as JSON string), band
    (safe / ambiguous / high_risk)"""
    df = df.copy()

    scores = []
    breakdowns = []
    bands = []

    for _, row in df.iterrows():
        result = score_transaction(row, df)
        score = result["raw_risk_score"]
        scores.append(score)
        breakdowns.append(json.dumps(result["rule_breakdown"]))

        if score < 30:
            bands.append("safe")
        elif score > 75:
            bands.append("high_risk")
        else:
            bands.append("ambiguous")

    df["raw_risk_score"] = scores
    df["rule_breakdown"] = breakdowns
    df["band"] = bands

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/transactions_train.csv")
    df["account_created_at"] = pd.to_datetime(df["account_created_at"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    scored = score_batch(df)

    print("=== Band Distribution ===")
    print(scored["band"].value_counts().to_string())
    print()

    ambiguous_fraud = scored[(scored["band"] == "ambiguous") & (scored["true_label"] == True)]
    safe_fraud = scored[(scored["band"] == "safe") & (scored["true_label"] == True)]
    ambiguous_fraud = scored[(scored["band"] == "ambiguous") & (scored["true_label"] == True)]
    highrisk_fraud = scored[(scored["band"] == "high_risk") & (scored["true_label"] == True)]
    print("=== Fraud Detection by Band ===")
    print(f"  safe (score < 30):      {len(safe_fraud)} fraud txns waved through with zero review")
    print(f"  ambiguous (30-75):      {len(ambiguous_fraud)} fraud txns escalated for AI review")
    print(f"  high_risk (score > 75): {len(highrisk_fraud)} fraud txns auto-blocked")
    if len(safe_fraud) > 0:
        print()
        print("  --- Fraud txns in safe band (slipping through) ---")
        for _, r in safe_fraud.iterrows():
            b = json.loads(r["rule_breakdown"])
            print(f"    {r['txn_id']}  amt={r['amount']:.0f}  score={r['raw_risk_score']}  breakdown={b}")
    print()

    print("=== Average Score by true_label ===")
    print(scored.groupby("true_label")["raw_risk_score"].mean().to_string())
    print()

    # --- Diagnostic: fraud sub-type breakdown ---
    fraud = scored[scored["true_label"] == True].copy()

    card_freq = fraud.groupby("card_fingerprint")["txn_id"].transform("count")
    card_testing = fraud[card_freq >= 4]
    ring_abuse = fraud[card_freq < 4]

    print("=== Fraud Sub-Type Average Scores ===")
    print(f"  Card-testing fraud (card appears 4+ times):  {len(card_testing)} txns, avg score = {card_testing['raw_risk_score'].mean():.2f}")
    print(f"  Ring-abuse fraud    (card appears <4 times):  {len(ring_abuse)} txns, avg score = {ring_abuse['raw_risk_score'].mean():.2f}")
    print()

    # --- Highest-scoring fraud transaction ---
    best_idx = fraud["raw_risk_score"].idxmax()
    best = fraud.loc[best_idx]
    import json
    breakdown = json.loads(best["rule_breakdown"])
    print("=== Highest-Scoring Fraud Transaction ===")
    print(f"  txn_id:          {best['txn_id']}")
    print(f"  card_fingerprint: {best['card_fingerprint']}")
    print(f"  amount:          {best['amount']}")
    print(f"  account_created: {best['account_created_at']}")
    print(f"  timestamp:       {best['timestamp']}")
    print(f"  raw_risk_score:  {best['raw_risk_score']}")
    print(f"  rule_breakdown:  {breakdown}")

    # --- Velocity debug: pick a fraud txn from a card-testing sequence ---
    fraud_df = scored[scored["true_label"] == True].copy()
    card_freq = fraud_df.groupby("card_fingerprint")["txn_id"].transform("count")
    card_test_txns = fraud_df[card_freq >= 2]
    if len(card_test_txns) > 0:
        target = card_test_txns.sort_values("timestamp").iloc[0]
    else:
        target = fraud_df.sort_values("raw_risk_score", ascending=False).iloc[0]
    target_tid = target["txn_id"]
    target_card = target["card_fingerprint"]
    target_ts = pd.Timestamp(target["timestamp"])
    same_card = scored[scored["card_fingerprint"] == target_card].copy()
    same_card["_ts"] = same_card["timestamp"].apply(lambda t: pd.Timestamp(t))
    same_card["_diff_min"] = same_card["_ts"].apply(lambda t: (t - target_ts).total_seconds() / 60.0)
    same_card["_in_window"] = same_card["_diff_min"].abs() <= 5
    before = same_card[(same_card["_diff_min"] < 0) & same_card["_in_window"]]
    after = same_card[(same_card["_diff_min"] > 0) & same_card["_in_window"]]
    at_self = same_card[same_card["_diff_min"] == 0]
    print()
    print(f"=== Velocity Debug: {target_tid} (card={target_card}, ts={target_ts}) ===")
    print(f"  Same-card txns total: {len(same_card)}")
    print(f"  Within 5min BEFORE:   {len(before)}")
    for _, r in before.iterrows():
        print(f"    {r['txn_id']}  ts={r['_ts']}  diff={r['_diff_min']:.1f}min  amt={r['amount']:.0f}")
    print(f"  At same timestamp:    {len(at_self)}")
    print(f"  Within 5min AFTER:    {len(after)}")
    for _, r in after.iterrows():
        print(f"    {r['txn_id']}  ts={r['_ts']}  diff={r['_diff_min']:.1f}min  amt={r['amount']:.0f}")
    print(f"  Total others in window: {len(before) + len(after)}")
    print(f"  velocity_score result:  {velocity_score(target, scored)}")
