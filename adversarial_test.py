"""
Adversarial Robustness Test — Probing the System's Blind Spots

Generates deliberately "near-miss" synthetic fraud cases engineered to sit
just below the system's detection thresholds, then reports how many slip
through undetected. This is an honest audit — if a meaningful fraction
escapes, we report it as a known limitation, not spin it as a win.

Thresholds pulled from engine/rules.py (lines 122-127):
  safe:       score < 30  (auto-approved, zero review)
  ambiguous:  30 <= score <= 75  (sent to Layer 2/3)
  high_risk:  score > 75  (auto-blocked)

Ring detection from engine/graph_builder.py (line 43):
  min_cluster_size = 3  (connected components with 3+ accounts)

NOTE: This script does NOT modify rules.py, graph_builder.py, or
ai_investigator.py. It only reads their thresholds and runs the existing
pipeline on crafted inputs.
"""

import json
import os
import random
import sys
from datetime import timedelta

import pandas as pd

random.seed(99)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad",
    "Chennai", "Kolkata", "Pune", "Jaipur", "Lucknow",
]


def _rand_id(prefix):
    import hashlib
    h = hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


# ---------------------------------------------------------------------------
# Pattern 1: Reduced-velocity card testing (near-miss for Layer 1)
# ---------------------------------------------------------------------------
def generate_reduced_velocity_card_testing(n=10):
    """Card-testing fraud with velocity tuned to land in ambiguous band.

    Normal card testing: 5+ small txns in 5min -> velocity=35 -> high_risk.
    Near-miss: 3-4 small txns -> velocity=21-28, keeping total < 75.

    With brand-new account (age=15) and geo mismatch (10):
      velocity=28 + age=15 + geo=10 = 53 -> ambiguous (30-75)
    Without geo mismatch:
      velocity=28 + age=15 = 43 -> ambiguous
    """
    rows = []
    txn_counter = 50000
    base_time = pd.Timestamp("2026-03-01 10:00:00")

    for seq in range(n):
        card = _rand_id("CARD")
        device = _rand_id("DEV")
        acct = _rand_id("ACCT")
        # Vary account age: some <10min (age=15), some 10-60min (age varies)
        age_minutes = random.choice([3, 5, 8, 15, 30, 45])
        created_at = base_time - timedelta(minutes=age_minutes)
        # Vary geo: some mismatch, some match
        billing_region = random.choice(INDIAN_CITIES)
        ip_region = billing_region if random.random() < 0.4 else random.choice(
            [c for c in INDIAN_CITIES if c != billing_region]
        )

        seq_start = base_time + timedelta(minutes=seq * 10)

        # 3-4 small test transactions (amount < 20, within 5min window)
        small_count = random.choice([3, 4])
        for j in range(small_count):
            ts = seq_start + timedelta(minutes=j * 0.8)
            rows.append({
                "txn_id": _rand_id("TXN"),
                "card_fingerprint": card,
                "device_id": device,
                "ip_region": ip_region,
                "account_id": acct,
                "account_created_at": created_at,
                "amount": round(random.uniform(5, 18), 2),
                "timestamp": ts,
                "billing_region": billing_region,
                "status": "approved",
                "decline_type": "",
                "true_label": True,
                "adversarial_pattern": "reduced_velocity_card_testing",
            })
            txn_counter += 1

        # Large target transaction
        large_ts = seq_start + timedelta(minutes=small_count * 0.8 + 0.5)
        rows.append({
            "txn_id": _rand_id("TXN"),
            "card_fingerprint": card,
            "device_id": device,
            "ip_region": ip_region,
            "account_id": acct,
            "account_created_at": created_at,
            "amount": round(random.uniform(3000, 12000), 2),
            "timestamp": large_ts,
            "billing_region": billing_region,
            "status": "approved",
            "decline_type": "",
            "true_label": True,
            "adversarial_pattern": "reduced_velocity_card_testing",
        })
        txn_counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pattern 2: Two-account ring (near-miss for Layer 2)
# ---------------------------------------------------------------------------
def generate_two_account_ring(n=8):
    """Ring abuse with only 2 accounts sharing a device — below the
    min_cluster_size=3 threshold, so Layer 2 won't flag it as a ring.

    Each account has 1-2 transactions. Combined with weak Layer 1 signals
    (new account, moderate amount) to land in ambiguous band.
    """
    rows = []
    relationships = []
    txn_counter = 51000
    base_time = pd.Timestamp("2026-03-02 10:00:00")

    for pair_idx in range(n):
        shared_device = _rand_id("DEV")
        acct_a = _rand_id("ACCT")
        acct_b = _rand_id("ACCT")

        # Relationship: shared device (but only 2 accounts -> not a ring)
        relationships.append({
            "account_id_a": acct_a,
            "account_id_b": acct_b,
            "shared_signal": "device",
        })

        created_a = base_time - timedelta(minutes=random.randint(5, 40))
        created_b = base_time - timedelta(minutes=random.randint(5, 40))

        for acct, created, card in [
            (acct_a, created_a, _rand_id("CARD")),
            (acct_b, created_b, _rand_id("CARD")),
        ]:
            # 1-2 transactions per account, moderate amounts, new accounts
            n_txns = random.choice([1, 2])
            for j in range(n_txns):
                ts = base_time + timedelta(minutes=pair_idx * 10 + j * 2)
                rows.append({
                    "txn_id": _rand_id("TXN"),
                    "card_fingerprint": card,
                    "device_id": shared_device,
                    "ip_region": random.choice(INDIAN_CITIES),
                    "account_id": acct,
                    "account_created_at": created,
                    "amount": round(random.uniform(2000, 8000), 2),
                    "timestamp": ts,
                    "billing_region": random.choice(INDIAN_CITIES),
                    "status": "approved",
                    "decline_type": "",
                    "true_label": True,
                    "adversarial_pattern": "two_account_ring",
                })
                txn_counter += 1

    rel_df = pd.DataFrame(relationships)
    return pd.DataFrame(rows), rel_df


# ---------------------------------------------------------------------------
# Pattern 3: Stacked weak signals (near-miss for Layer 1)
# ---------------------------------------------------------------------------
def generate_stacked_weak_signals(n=10):
    """Fraud with multiple weak signals that each contribute a few points,
    stacking to land in the ambiguous band (30-75) but never high_risk (>75).

    Strategy: account_age (partial) + geo_mismatch (10) + moderate velocity
    but NOT enough for amount_pattern (no small-prior-txns).

    Examples:
      age=10 + geo=10 + velocity=14 = 34 -> just above safe threshold
      age=15 + geo=10 + velocity=7 = 32 -> barely ambiguous
      age=8 + geo=10 + velocity=14 = 32 -> barely ambiguous
    """
    rows = []
    txn_counter = 52000
    base_time = pd.Timestamp("2026-03-03 10:00:00")

    for seq in range(n):
        card = _rand_id("CARD")
        device = _rand_id("DEV")
        acct = _rand_id("ACCT")

        # Vary account age: 10-50 minutes (partial points)
        age_minutes = random.randint(10, 50)
        created_at = base_time - timedelta(minutes=age_minutes)

        billing = random.choice(INDIAN_CITIES)
        ip = random.choice([c for c in INDIAN_CITIES if c != billing])

        seq_start = base_time + timedelta(minutes=seq * 8)

        # 2-3 same-card txns in the window (velocity: 14-21)
        prior_count = random.choice([2, 3])
        for j in range(prior_count):
            ts = seq_start + timedelta(minutes=j * 1.5)
            rows.append({
                "txn_id": _rand_id("TXN"),
                "card_fingerprint": card,
                "device_id": device,
                "ip_region": ip,
                "account_id": acct,
                "account_created_at": created_at,
                "amount": round(random.uniform(100, 500), 2),  # >20, so no amount_pattern
                "timestamp": ts,
                "billing_region": billing,
                "status": "approved",
                "decline_type": "",
                "true_label": True,
                "adversarial_pattern": "stacked_weak_signals",
            })
            txn_counter += 1

        # Target transaction
        target_ts = seq_start + timedelta(minutes=prior_count * 1.5 + 1)
        rows.append({
            "txn_id": _rand_id("TXN"),
            "card_fingerprint": card,
            "device_id": device,
            "ip_region": ip,
            "account_id": acct,
            "account_created_at": created_at,
            "amount": round(random.uniform(3000, 10000), 2),
            "timestamp": target_ts,
            "billing_region": billing,
            "status": "approved",
            "decline_type": "",
            "true_label": True,
            "adversarial_pattern": "stacked_weak_signals",
        })
        txn_counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pattern 4: Isolated new-account fraud (near-miss for Layer 1)
# ---------------------------------------------------------------------------
def generate_isolated_new_account(n=10):
    """Single-transaction fraud from a brand-new account with geo mismatch.
    No velocity signal (single txn), no amount pattern (no prior txns).

    Score: age(15) + geo(10) = 25 -> safe band (auto-approved).
    This is a TRUE blind spot: a brand-new account making a single large
    fraudulent purchase from a different region scores below 30.

    Adding a small amount pattern trigger would push it to 55 (ambiguous),
    but the attacker deliberately avoids small test transactions.
    """
    rows = []
    txn_counter = 53000
    base_time = pd.Timestamp("2026-03-04 10:00:00")

    for i in range(n):
        acct = _rand_id("ACCT")
        card = _rand_id("CARD")
        device = _rand_id("DEV")
        # Brand new account (2-8 minutes old)
        age_minutes = random.randint(2, 8)
        created_at = base_time - timedelta(minutes=age_minutes)
        billing = random.choice(INDIAN_CITIES)
        ip = random.choice([c for c in INDIAN_CITIES if c != billing])

        ts = base_time + timedelta(minutes=i * 5)
        rows.append({
            "txn_id": _rand_id("TXN"),
            "card_fingerprint": card,
            "device_id": device,
            "ip_region": ip,
            "account_id": acct,
            "account_created_at": created_at,
            "amount": round(random.uniform(5000, 15000), 2),
            "timestamp": ts,
            "billing_region": billing,
            "status": "approved",
            "decline_type": "",
            "true_label": True,
            "adversarial_pattern": "isolated_new_account",
        })
        txn_counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline runner (Layers 1+2 only, no AI calls)
# ---------------------------------------------------------------------------
def run_pipeline_layers_1_2(adversarial_df, test_df, relationships_path):
    """Run Layer 1 (rules) and Layer 2 (graph) on combined data.
    Returns the adversarial rows with scores and bands attached."""
    from engine import rules, graph_builder

    # Combine adversarial with existing test data
    combined = pd.concat([test_df, adversarial_df], ignore_index=True)
    combined["account_created_at"] = pd.to_datetime(combined["account_created_at"])
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    # Layer 1: Rule scoring
    combined = rules.score_batch(combined)

    # Layer 2: Graph / ring detection
    # Append adversarial relationships to the relationships file
    adv_rels_path = os.path.join(RESULTS_DIR, "adversarial_relationships.csv")
    if os.path.exists(adv_rels_path):
        adv_rels = pd.read_csv(adv_rels_path)
        orig_rels = pd.read_csv(relationships_path)
        combined_rels = pd.concat([orig_rels, adv_rels], ignore_index=True)
        combined_rels.to_csv(adv_rels_path + ".tmp", index=False)
        graph = graph_builder.build_graph(combined)
        os.remove(adv_rels_path + ".tmp")
    else:
        graph = graph_builder.build_graph(combined)

    ring_candidates = graph_builder.find_ring_candidates(graph)
    all_ring_accounts = set()
    for ring in ring_candidates:
        all_ring_accounts |= ring
    combined["in_detected_ring"] = combined["account_id"].isin(all_ring_accounts)

    # Extract adversarial rows
    adv_txn_ids = set(adversarial_df["txn_id"])
    adv_results = combined[combined["txn_id"].isin(adv_txn_ids)].copy()

    return adv_results, ring_candidates


def classify_verdicts(adv_results):
    """Classify each adversarial transaction into outcome categories.

    Categories:
      - auto_blocked: high_risk band -> fraud (best outcome for detection)
      - manual_review: ambiguous band -> sent to human review (good outcome)
      - auto_approved: safe band -> legitimate (MISS — fraud went undetected)
    """
    outcomes = []
    for _, row in adv_results.iterrows():
        band = row["band"]
        if band == "high_risk":
            outcome = "auto_blocked"
        elif band == "ambiguous":
            outcome = "manual_review"
        else:
            outcome = "auto_approved"
        outcomes.append(outcome)
    adv_results["outcome"] = outcomes
    return adv_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from engine import rules

    print("=" * 70)
    print("ADVERSARIAL ROBUSTNESS TEST")
    print("Probing the system's blind spots with near-miss fraud cases")
    print("=" * 70)
    print()

    # Report current thresholds
    print("Current detection thresholds (from engine/rules.py):")
    print("  safe:       score < 30  (auto-approved)")
    print("  ambiguous:  30 <= score <= 75  (AI review)")
    print("  high_risk:  score > 75  (auto-blocked)")
    print("  Ring detection: min_cluster_size = 3 (engine/graph_builder.py)")
    print()

    # Load existing test set
    test_path = os.path.join(DATA_DIR, "transactions_test.csv")
    if not os.path.exists(test_path):
        print("ERROR: transactions_test.csv not found. Run data/generate_dataset.py first.")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    test_df["account_created_at"] = pd.to_datetime(test_df["account_created_at"])
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    print(f"Loaded held-out test set: {len(test_df)} transactions")
    print()

    # Generate adversarial patterns
    print("Generating near-miss fraud cases...")
    p1 = generate_reduced_velocity_card_testing(n=10)
    p2_df, p2_rels = generate_two_account_ring(n=8)
    p3 = generate_stacked_weak_signals(n=10)
    p4 = generate_isolated_new_account(n=10)

    # Save relationships for pattern 2 (two-account rings)
    rels_path = os.path.join(RESULTS_DIR, "adversarial_relationships.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p2_rels.to_csv(rels_path, index=False)
    print(f"  Wrote {len(p2_rels)} adversarial relationships to {rels_path}")

    # Combine all patterns
    adversarial_df = pd.concat([p1, p2_df, p3, p4], ignore_index=True)
    adversarial_df["amount"] = adversarial_df["amount"].astype(float)
    adversarial_df["true_label"] = adversarial_df["true_label"].astype(bool)

    print(f"  Generated {len(adversarial_df)} near-miss fraud transactions:")
    print(f"    Pattern 1 - Reduced velocity card testing: {len(p1)} txns")
    print(f"    Pattern 2 - Two-account ring:              {len(p2_df)} txns")
    print(f"    Pattern 3 - Stacked weak signals:          {len(p3)} txns")
    print(f"    Pattern 4 - Isolated new account:          {len(p4)} txns")
    print()

    # Run through pipeline (Layers 1+2 only, no AI calls)
    print("Running through pipeline (Layer 1 rules + Layer 2 graph)...")
    adv_results, ring_candidates = run_pipeline_layers_1_2(
        adversarial_df, test_df, os.path.join(DATA_DIR, "account_relationships.csv")
    )

    # Classify outcomes
    adv_results = classify_verdicts(adv_results)

    # Summary
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    total = len(adv_results)
    blocked = (adv_results["outcome"] == "auto_blocked").sum()
    review = (adv_results["outcome"] == "manual_review").sum()
    missed = (adv_results["outcome"] == "auto_approved").sum()

    print(f"Total near-miss fraud cases tested: {total}")
    print(f"  Auto-blocked (high_risk):     {blocked:>3}  ({blocked/total*100:.1f}%)")
    print(f"  Sent to manual review:        {review:>3}  ({review/total*100:.1f}%)")
    print(f"  AUTO-APPROVED (MISSED):       {missed:>3}  ({missed/total*100:.1f}%)")
    print()

    caught = blocked + review
    print(f"  Caught (block + review):      {caught:>3}  ({caught/total*100:.1f}%)")
    print(f"  Escaped detection:            {missed:>3}  ({missed/total*100:.1f}%)")
    print()

    # Per-pattern breakdown
    print("--- Breakdown by Fraud Pattern ---")
    print()
    for pattern in adversarial_df["adversarial_pattern"].unique():
        pat_results = adv_results[adv_results["adversarial_pattern"] == pattern]
        pat_total = len(pat_results)
        pat_blocked = (pat_results["outcome"] == "auto_blocked").sum()
        pat_review = (pat_results["outcome"] == "manual_review").sum()
        pat_missed = (pat_results["outcome"] == "auto_approved").sum()

        print(f"  {pattern}:")
        print(f"    Total: {pat_total}, Blocked: {pat_blocked}, Review: {pat_review}, Missed: {pat_missed}")

        # Show score distribution for missed cases
        missed_rows = pat_results[pat_results["outcome"] == "auto_approved"]
        if len(missed_rows) > 0:
            scores = missed_rows["raw_risk_score"].tolist()
            print(f"    Missed scores: {scores} (all < 30 = safe band)")

        # Show score distribution for review cases
        review_rows = pat_results[pat_results["outcome"] == "manual_review"]
        if len(review_rows) > 0:
            scores = review_rows["raw_risk_score"].tolist()
            print(f"    Review scores: {scores} (30-75 = ambiguous band)")
        print()

    # Score distribution overview
    print("--- Score Distribution (all near-miss cases) ---")
    print(f"  Mean score:  {adv_results['raw_risk_score'].mean():.1f}")
    print(f"  Min score:   {adv_results['raw_risk_score'].min()}")
    print(f"  Max score:   {adv_results['raw_risk_score'].max()}")
    print(f"  Median:      {adv_results['raw_risk_score'].median():.1f}")
    print()

    # Band distribution
    band_dist = adv_results["band"].value_counts()
    print("  Band distribution:")
    for band in ["safe", "ambiguous", "high_risk"]:
        count = band_dist.get(band, 0)
        print(f"    {band:>10}: {count:>3} ({count/total*100:.1f}%)")
    print()

    # Honest assessment
    print("=" * 70)
    print("HONEST ASSESSMENT")
    print("=" * 70)
    print()
    if missed == 0:
        print("  All near-miss cases were caught (blocked or sent to review).")
        print("  The system's thresholds appear robust to the tested attack patterns.")
    else:
        print(f"  WARNING: {missed}/{total} near-miss fraud cases ({missed/total*100:.1f}%)")
        print(f"  were AUTO-APPROVED with zero human review.")
        print()
        print("  This means a determined attacker who understands the scoring")
        print("  rules could craft transactions that land in the safe band (<30)")
        print("  and bypass all detection layers.")
        print()
        if missed > 0:
            missed_patterns = adv_results[adv_results["outcome"] == "auto_approved"][
                "adversarial_pattern"
            ].value_counts()
            print("  Most vulnerable patterns:")
            for pat, count in missed_patterns.items():
                print(f"    - {pat}: {count} cases missed")
    print()

    # Ring detection check
    ring_flagged = adv_results["in_detected_ring"].sum()
    print(f"  Ring detection: {ring_flagged}/{total} adversarial cases flagged by Layer 2")
    two_acct_ring = adv_results[adv_results["adversarial_pattern"] == "two_account_ring"]
    if len(two_acct_ring) > 0:
        two_ring_flagged = two_acct_ring["in_detected_ring"].sum()
        print(f"    Two-account ring cases flagged: {two_ring_flagged}/{len(two_acct_ring)}")
        print(f"    (Expected: 0 — 2-account rings are below min_cluster_size=3)")
    print()

    # Save results
    results = {
        "test_description": "Adversarial robustness test with near-miss fraud cases",
        "thresholds": {
            "safe_below": 30,
            "ambiguous_range": "30-75",
            "high_risk_above": 75,
            "ring_min_cluster_size": 3,
        },
        "total_cases": total,
        "outcomes": {
            "auto_blocked": int(blocked),
            "manual_review": int(review),
            "auto_approved_missed": int(missed),
        },
        "percentages": {
            "auto_blocked_pct": round(blocked / total * 100, 1),
            "manual_review_pct": round(review / total * 100, 1),
            "missed_pct": round(missed / total * 100, 1),
        },
        "pattern_breakdown": {},
        "score_stats": {
            "mean": round(float(adv_results["raw_risk_score"].mean()), 1),
            "min": int(adv_results["raw_risk_score"].min()),
            "max": int(adv_results["raw_risk_score"].max()),
        },
        "ring_detection": {
            "total_flagged": int(ring_flagged),
            "two_account_ring_flagged": int(two_acct_ring["in_detected_ring"].sum()) if len(two_acct_ring) > 0 else 0,
        },
    }

    for pattern in adv_results["adversarial_pattern"].unique():
        pat = adv_results[adv_results["adversarial_pattern"] == pattern]
        results["pattern_breakdown"][pattern] = {
            "total": int(len(pat)),
            "auto_blocked": int((pat["outcome"] == "auto_blocked").sum()),
            "manual_review": int((pat["outcome"] == "manual_review").sum()),
            "auto_approved_missed": int((pat["outcome"] == "auto_approved").sum()),
        }

    results_path = os.path.join(RESULTS_DIR, "adversarial_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Also save detailed per-transaction results
    detail_path = os.path.join(RESULTS_DIR, "adversarial_details.csv")
    detail_cols = [
        "txn_id", "adversarial_pattern", "amount", "raw_risk_score", "band",
        "outcome", "in_detected_ring",
    ]
    adv_results[detail_cols].to_csv(detail_path, index=False)
    print(f"Per-transaction details saved to {detail_path}")

    return results


if __name__ == "__main__":
    main()
