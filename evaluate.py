"""
DAY 6 — Evaluation + Confidence Calibration (DO NOT SKIP/SHORTEN)

Runs the FULL pipeline (rules -> graph -> AI investigator -> fork) on
the FROZEN transactions_test.csv (untouched since Day 1).

Computes:
  - precision, recall, false-positive rate, false-negative rate
  - ring-detection accuracy vs account_relationships.csv ground truth
  - confidence calibration table (stated confidence bucket vs actual
    accuracy in that bucket — proves confidence numbers are earned)
  - recovery metrics: recovered / at-risk, split soft vs hard decline

Writes:
  results/evaluation_report.json
  results/evaluation_report.md
  results/calibration_table.csv
"""

import json
import os
import re
import sys
import time

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from engine import rules, graph_builder, ai_investigator, recovery


def run_full_pipeline(df: pd.DataFrame, relationships_path: str = "data/account_relationships.csv",
                      cache_only: bool = False) -> pd.DataFrame:
    """Run every layer end to end on the given dataframe, return
    df with added columns: raw_risk_score, verdict, confidence,
    recommended_action, degraded_mode, etc.

    If cache_only=True, skip ambiguous transactions that have no cached
    AI investigation result — do not call the API.  Skipped transactions
    are recorded as ai_verdict='skipped_no_cache' so the report clearly
    states how many were partial."""
    df = df.copy()

    # --- Layer 1: Rule Engine ---
    df = rules.score_batch(df)

    # --- Layer 2: Graph / Ring Detection ---
    graph = graph_builder.build_graph(df)
    ring_candidates = graph_builder.find_ring_candidates(graph)

    all_ring_accounts = set()
    account_to_ring = {}
    for ring in ring_candidates:
        ring_id = frozenset(ring)
        for acct in ring:
            all_ring_accounts.add(acct)
            account_to_ring[acct] = ring_id

    df["in_detected_ring"] = df["account_id"].isin(all_ring_accounts)

    # Build relationships lookup for graph_info passed to AI investigator
    if os.path.exists(relationships_path):
        relationships_df = pd.read_csv(relationships_path)
    else:
        print(
            "WARNING: account_relationships.csv not found. "
            "Ring-aware graph_info will not be passed to AI investigator.",
            file=sys.stderr,
        )
        relationships_df = pd.DataFrame(columns=["account_id_a", "account_id_b", "shared_signal"])

    # --- Layer 3: AI Investigator (ambiguous band only) ---
    ambiguous_mask = df["band"] == "ambiguous"
    ambiguous_indices = df[ambiguous_mask].index

    ai_verdicts = [None] * len(df)
    ai_confidences = [None] * len(df)
    ai_reasonings = [None] * len(df)
    ai_actions = [None] * len(df)
    degraded_modes = [False] * len(df)
    skipped_no_cache = []

    for i, idx in enumerate(ambiguous_indices):
        row = df.loc[idx]
        rule_breakdown = json.loads(row["rule_breakdown"]) if isinstance(row["rule_breakdown"], str) else row["rule_breakdown"]

        # --- Cache-only mode: check cache before calling API ---
        if cache_only:
            cache_path = os.path.join("cache", f"investigate_{row['txn_id']}.json")
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    cached = json.load(f)
                ai_verdicts[idx] = cached.get("verdict")
                ai_confidences[idx] = cached.get("confidence")
                ai_reasonings[idx] = cached.get("reasoning")
                ai_actions[idx] = cached.get("recommended_action")
                degraded_modes[idx] = cached.get("degraded_mode", False)
                continue
            skipped_no_cache.append(row['txn_id'])
            ai_verdicts[idx] = "skipped_no_cache"
            ai_confidences[idx] = None
            ai_reasonings[idx] = "Skipped: no cached AI result available (--cache-only mode)"
            ai_actions[idx] = "manual_review"
            degraded_modes[idx] = False
            print(f"  SKIP (no cache): {row['txn_id']}", file=sys.stderr)
            continue

        # Build graph_info if account is in a detected ring
        graph_info = None
        if row["in_detected_ring"]:
            ring_id = account_to_ring[row["account_id"]]
            connected = relationships_df[
                (relationships_df["account_id_a"] == row["account_id"]) |
                (relationships_df["account_id_b"] == row["account_id"])
            ]
            connected_accounts = []
            for _, rel in connected.iterrows():
                other = rel["account_id_b"] if rel["account_id_a"] == row["account_id"] else rel["account_id_a"]
                connected_accounts.append({
                    "account_id": other,
                    "shared_signal": rel["shared_signal"],
                })
            graph_info = {"connected_accounts": connected_accounts}

        txn_dict = row.to_dict()

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = ai_investigator.investigate(txn_dict, rule_breakdown, graph_info)
                break
            except Exception as e:
                err_str = str(e)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str.upper() or "rate" in err_str.lower()
                if is_429 and attempt < max_attempts - 1:
                    match = re.search(r"retry[_\s=:]*(?:delay)[:\s=]*(\d+)", err_str, re.IGNORECASE)
                    delay = int(match.group(1)) if match else 30
                    print(f"429 RATE LIMITED on {row['txn_id']}: retryDelay={delay}s (attempt {attempt+1}/{max_attempts})", file=sys.stderr)
                    time.sleep(delay + 2)
                else:
                    print(f"INVESTIGATE FAILED (attempt {attempt+1}): {type(e).__name__}: {e}", file=sys.stderr)
                    result = {
                        "verdict": "insufficient_evidence",
                        "confidence": None,
                        "reasoning": f"API error: {e}",
                        "recommended_action": "manual_review",
                        "degraded_mode": True,
                    }
                    break

        ai_verdicts[idx] = result.get("verdict")
        ai_confidences[idx] = result.get("confidence")
        ai_reasonings[idx] = result.get("reasoning")
        ai_actions[idx] = result.get("recommended_action")
        degraded_modes[idx] = result.get("degraded_mode", False)

        if i < len(ambiguous_indices) - 1:
            time.sleep(2)

    df["ai_verdict"] = ai_verdicts
    df["ai_confidence"] = ai_confidences
    df["ai_reasoning"] = ai_reasonings
    df["ai_recommended_action"] = ai_actions
    df["degraded_mode"] = degraded_modes
    df.attrs["skipped_no_cache"] = skipped_no_cache
    df.attrs["cache_only_mode"] = cache_only

    # --- Final Decision Logic ---
    def decide_final(row):
        if row["band"] == "high_risk":
            return "fraud"
        if row["band"] == "safe":
            return "legitimate"
        if row["band"] == "ambiguous":
            verdict = row["ai_verdict"]
            confidence = row["ai_confidence"]
            if verdict == "fraud_likely" and confidence is not None and confidence >= 65:
                return "fraud"
            if verdict == "legitimate_likely" and confidence is not None and confidence >= 65:
                return "legitimate"
            return "manual_review"
        return "manual_review"

    df["final_decision"] = df.apply(decide_final, axis=1)

    # --- Revenue Recovery on declined non-fraud ---
    declined_mask = (df["status"] == "declined") & (df["final_decision"] != "fraud")
    declined_df = df[declined_mask].copy()

    recovery_results = {}
    recovery_per_txn = {}
    if len(declined_df) > 0:
        recovery_results = recovery.run_recovery_batch(declined_df)
        for entry in recovery_results.get("per_transaction_log", []):
            recovery_per_txn[entry["txn_id"]] = entry

    df["recovery_route"] = df["txn_id"].map(
        lambda tid: recovery_per_txn.get(tid, {}).get("route")
    )
    df["recovery_recovered"] = df["txn_id"].map(
        lambda tid: recovery_per_txn.get(tid, {}).get("recovered")
    )
    df["recovery_amount"] = df["txn_id"].map(
        lambda tid: recovery_per_txn.get(tid, {}).get("recovered_amount", 0.0)
    )

    df.attrs["recovery_summary"] = recovery_results

    return df


def compute_classification_metrics(results_df: pd.DataFrame) -> dict:
    """precision, recall, false_positive_rate, false_negative_rate
    against true_label column."""
    eval_df = results_df[results_df["final_decision"] != "manual_review"].copy()

    y_true = eval_df["true_label"].astype(int)
    y_pred = (eval_df["final_decision"] == "fraud").astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    manual_review_count = int((results_df["final_decision"] == "manual_review").sum())
    total = len(results_df)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "manual_review_count": manual_review_count,
        "manual_review_pct": round(manual_review_count / total, 4) if total > 0 else 0.0,
        "evaluated_count": len(eval_df),
        "total_count": total,
    }


def compute_fraud_typology(results_df: pd.DataFrame) -> dict:
    """Break down true-positive fraud by type: card-testing vs ring-abuse.

    Fraud type is inferred from the generator's patterns:
    - Card-testing: same card_fingerprint appears in multiple fraud txns
    - Ring-abuse: card_fingerprint appears in only one fraud txn (accounts
      share devices/addresses, not cards)

    Also reports overlap with Layer 2 ring-flagged status.

    Returns dict with per-type counts, amounts, and ring overlap."""
    fraud_df = results_df[results_df["true_label"] == True].copy()
    if len(fraud_df) == 0:
        return {"types": {}, "ring_overlap": {}, "total_fraud": 0}

    # Infer fraud type from card_fingerprint frequency in fraud txns
    card_fraud_counts = fraud_df.groupby("card_fingerprint").size()
    card_testing_cards = set(card_fraud_counts[card_fraud_counts > 1].index)

    fraud_df["fraud_type"] = fraud_df["card_fingerprint"].apply(
        lambda c: "card_testing" if c in card_testing_cards else "ring_abuse"
    )

    types = {}
    for ftype in ["card_testing", "ring_abuse"]:
        subset = fraud_df[fraud_df["fraud_type"] == ftype]
        types[ftype] = {
            "count": int(len(subset)),
            "total_amount": round(float(subset["amount"].sum()), 2),
        }

    # Ring overlap: how many of each type were also flagged by Layer 2
    ring_overlap = {}
    if "in_detected_ring" in fraud_df.columns:
        for ftype in ["card_testing", "ring_abuse"]:
            subset = fraud_df[fraud_df["fraud_type"] == ftype]
            ring_count = int(subset["in_detected_ring"].sum())
            ring_overlap[ftype] = {
                "ring_flagged": ring_count,
                "total": int(len(subset)),
            }

    return {
        "types": types,
        "ring_overlap": ring_overlap,
        "total_fraud": int(len(fraud_df)),
    }


def compute_calibration_table(results_df: pd.DataFrame, n_buckets=5) -> pd.DataFrame:
    """Bucket by stated confidence (0-20, 20-40, ..., 80-100),
    compute actual accuracy within each bucket."""
    ai_rows = results_df[results_df["ai_confidence"].notna()].copy()

    if len(ai_rows) == 0:
        return pd.DataFrame(columns=["confidence_bucket", "count", "actual_accuracy"])

    bins = [i * (100 // n_buckets) for i in range(n_buckets + 1)]
    bins[-1] = 100.1
    labels = []
    for i in range(n_buckets):
        lo = i * (100 // n_buckets)
        hi = (i + 1) * (100 // n_buckets)
        labels.append(f"{lo}-{hi}")

    ai_rows["confidence_bucket"] = pd.cut(
        ai_rows["ai_confidence"],
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []
    for bucket in labels:
        bucket_rows = ai_rows[ai_rows["confidence_bucket"] == bucket]
        count = len(bucket_rows)
        if count == 0:
            rows.append({
                "confidence_bucket": bucket,
                "count": 0,
                "actual_accuracy": None,
            })
            continue

        correct = 0
        for _, r in bucket_rows.iterrows():
            ai_says_fraud = r["ai_verdict"] == "fraud_likely"
            truth_is_fraud = r["true_label"] == True
            if ai_says_fraud == truth_is_fraud:
                correct += 1

        rows.append({
            "confidence_bucket": bucket,
            "count": count,
            "actual_accuracy": round(correct / count, 4),
        })

    return pd.DataFrame(rows)


def compute_ring_accuracy(results_df, relationships_df) -> dict:
    """Compare detected rings against ground-truth relationships."""
    if relationships_df is None or relationships_df.empty:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "detected_ring_accounts": 0,
            "ground_truth_ring_accounts": 0,
            "note": "account_relationships.csv not found — ring accuracy unavailable",
        }
    ground_truth_accounts = set(relationships_df["account_id_a"].unique()) | \
                            set(relationships_df["account_id_b"].unique())

    detected_ring_accounts = set(results_df[results_df["in_detected_ring"]]["account_id"].unique())
    all_accounts = set(results_df["account_id"].unique())

    true_positives = detected_ring_accounts & ground_truth_accounts
    false_positives = detected_ring_accounts - ground_truth_accounts
    false_negatives = ground_truth_accounts - all_accounts - detected_ring_accounts
    # false_negatives that are in the test set
    false_negatives_in_test = (ground_truth_accounts & all_accounts) - detected_ring_accounts

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives_in_test)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "detected_ring_accounts": len(detected_ring_accounts),
        "ground_truth_ring_accounts": len(ground_truth_accounts & all_accounts),
    }


def compute_recovery_metrics(results_df) -> dict:
    """Pull recovery results, split by soft/hard decline."""
    summary = results_df.attrs.get("recovery_summary", {})

    total_at_risk = summary.get("total_at_risk", 0.0)
    total_recovered = summary.get("total_recovered", 0.0)
    recovery_rate = (total_recovered / total_at_risk * 100) if total_at_risk > 0 else 0.0

    return {
        "total_at_risk": round(total_at_risk, 2),
        "total_recovered": round(total_recovered, 2),
        "recovery_rate_pct": round(recovery_rate, 2),
        "soft_decline_recovered": round(summary.get("soft_decline_recovered", 0.0), 2),
        "hard_decline_recovered": round(summary.get("hard_decline_recovered", 0.0), 2),
        "soft_decline_count": summary.get("soft_decline_count", 0),
        "hard_decline_count": summary.get("hard_decline_count", 0),
        "suppressed_count": summary.get("suppressed_count", 0),
    }


def compute_multi_run_recall(n_runs: int = 5, test_df: pd.DataFrame = None,
                             relationships_path: str = "data/account_relationships.csv",
                             cache_only: bool = False) -> dict:
    """Run the full pipeline N times and report recall as a range.

    If cache_only=False (default): clears cache between runs to capture
    Gemini API non-determinism — this is the original behavior.

    If cache_only=True: does NOT clear cache between runs — all runs
    reuse the same cached results.  Useful for demo-safe partial eval.

    Returns dict with per_run recall values, min, max, mean, and note.
    """
    import shutil

    if test_df is None:
        test_df = pd.read_csv("data/transactions_test.csv")

    recalls = []
    tps = []
    fns = []
    skipped_counts = []
    for i in range(n_runs):
        # Only clear cache if NOT in cache-only mode
        if not cache_only:
            cache_dir = "cache"
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir)

        run_df = run_full_pipeline(test_df, relationships_path, cache_only=cache_only)
        skipped = run_df.attrs.get("skipped_no_cache", [])
        skipped_counts.append(len(skipped))
        metrics = compute_classification_metrics(run_df)
        recalls.append(metrics["recall"])
        tps.append(metrics["tp"])
        fns.append(metrics["fn"])

    recall_min = min(recalls)
    recall_max = max(recalls)
    recall_mean = sum(recalls) / len(recalls)

    return {
        "num_runs": n_runs,
        "recall_values": [round(r, 4) for r in recalls],
        "recall_min": round(recall_min, 4),
        "recall_max": round(recall_max, 4),
        "recall_mean": round(recall_mean, 4),
        "tp_values": tps,
        "fn_values": fns,
        "skipped_per_run": skipped_counts,
        "cache_only_mode": cache_only,
        "note": (
            "AI-investigator recall varies across runs due to non-determinism "
            "in the Gemini API despite temperature=0.0/seed=42. Only "
            "ambiguous-band (AI-investigated) cases contribute to variance; "
            "Layer 1 (rules) and Layer 2 (graph) verdicts are 100% stable. "
            + ("Cache-only mode: reused cached results across all runs. "
               "Skipped counts show how many ambiguous txns had no cached result."
               if cache_only else
               "Cache was cleared between runs to force fresh API calls. "
               "See diagnose_nondeterminism.py for per-transaction analysis.")
        ),
    }


def write_reports(all_metrics: dict):
    """Save JSON + human-readable Markdown report to results/"""
    os.makedirs("results", exist_ok=True)

    # --- Calibration sample-size warnings ---
    cal = all_metrics.get("calibration", {})
    small_sample_warnings = []
    if isinstance(cal, dict) and "confidence_bucket" in cal:
        buckets_raw = cal.get("confidence_bucket", [])
        counts_raw = cal.get("count", [])
        for b, cnt in zip(buckets_raw, counts_raw):
            if cnt is not None and 0 < cnt < 5:
                small_sample_warnings.append(
                    f"Bucket '{b}' has only {cnt} samples — calibration "
                    f"estimate has low statistical confidence."
                )
    all_metrics["calibration_sample_size_warnings"] = small_sample_warnings

    # --- JSON report ---
    with open("results/evaluation_report.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # --- Calibration table CSV ---
    cal_data = all_metrics.get("calibration", {})
    if isinstance(cal_data, dict) and "confidence_bucket" in cal_data:
        cal_df = pd.DataFrame(cal_data)
    elif isinstance(cal_data, pd.DataFrame):
        cal_df = cal_data
    else:
        cal_df = pd.DataFrame()
    if len(cal_df) > 0:
        cal_df.to_csv("results/calibration_table.csv", index=False)

    # --- Markdown report ---
    c = all_metrics.get("classification", {})
    r = all_metrics.get("recovery", {})
    ring = all_metrics.get("ring_detection", {})
    cal = all_metrics.get("calibration", {})

    md = []
    md.append("# Sentinel — Evaluation Report\n")

    # Cache-only mode banner
    if all_metrics.get("cache_only_mode"):
        skipped_count = all_metrics.get("skipped_no_cache_count", 0)
        skipped_txns = all_metrics.get("skipped_no_cache_txns", [])
        md.append("> **[CACHE-ONLY MODE]** This report was generated using only cached AI results. "
                  f"{skipped_count} ambiguous transaction(s) had no cached result and were skipped. "
                  "Metrics are partial, not complete.\n")
        if skipped_txns:
            md.append(f"> Skipped transactions: {', '.join(skipped_txns)}\n")
        md.append("")

    md.append("## Classification Metrics\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Precision | {c.get('precision', 'N/A')} |")
    md.append(f"| Recall | {c.get('recall', 'N/A')} |")
    md.append(f"| F1 Score | {c.get('f1_score', 'N/A')} |")
    md.append(f"| False Positive Rate | {c.get('false_positive_rate', 'N/A')} |")
    md.append(f"| False Negative Rate | {c.get('false_negative_rate', 'N/A')} |")
    md.append(f"| TP / FP / TN / FN | {c.get('tp',0)} / {c.get('fp',0)} / {c.get('tn',0)} / {c.get('fn',0)} |")
    md.append(f"| Manual Review Count | {c.get('manual_review_count', 0)} ({c.get('manual_review_pct', 0):.1%}) |")
    md.append(f"| Evaluated (excl. manual_review) | {c.get('evaluated_count', 0)} / {c.get('total_count', 0)} |")
    md.append("")

    # Fraud Typology Breakdown
    ft = all_metrics.get("fraud_typology", {})
    if ft and ft.get("types"):
        md.append("## Fraud Typology Breakdown\n")
        md.append("| Type | Count | Total Amount (INR) | Ring-Flagged |")
        md.append("|------|-------|-------------------|--------------|")
        for ftype in ["card_testing", "ring_abuse"]:
            type_data = ft["types"].get(ftype, {})
            ring_data = ft.get("ring_overlap", {}).get(ftype, {})
            count = type_data.get("count", 0)
            amt = type_data.get("total_amount", 0)
            ring_flagged = ring_data.get("ring_flagged", 0)
            ring_total = ring_data.get("total", 0)
            ring_str = f"{ring_flagged}/{ring_total}" if ring_total > 0 else "N/A"
            md.append(f"| {ftype.replace('_', ' ').title()} | {count} | {amt:,.2f} | {ring_str} |")
        md.append(f"| **Total** | **{ft.get('total_fraud', 0)}** | | |")
        md.append("")

    md.append("## Ring Detection Accuracy\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Precision | {ring.get('precision', 'N/A')} |")
    md.append(f"| Recall | {ring.get('recall', 'N/A')} |")
    md.append(f"| F1 Score | {ring.get('f1_score', 'N/A')} |")
    md.append(f"| Detected Ring Accounts | {ring.get('detected_ring_accounts', 0)} |")
    md.append(f"| Ground Truth Ring Accounts (in test set) | {ring.get('ground_truth_ring_accounts', 0)} |")
    md.append(f"| True Positives / False Positives / False Negatives | {ring.get('true_positives',0)} / {ring.get('false_positives',0)} / {ring.get('false_negatives',0)} |")
    md.append("")

    md.append("## Revenue Recovery\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Total At Risk | INR {r.get('total_at_risk', 0):,.2f} |")
    md.append(f"| Total Recovered | INR {r.get('total_recovered', 0):,.2f} |")
    md.append(f"| Recovery Rate | {r.get('recovery_rate_pct', 0):.1f}% |")
    md.append(f"| Soft Decline Recovered | INR {r.get('soft_decline_recovered', 0):,.2f} |")
    md.append(f"| Hard Decline Recovered | INR {r.get('hard_decline_recovered', 0):,.2f} |")
    md.append(f"| Suppressed Count | {r.get('suppressed_count', 0)} |")
    md.append("")

    md.append("## Confidence Calibration Table\n")
    md.append(f"| Bucket | Count | Actual Accuracy |")
    md.append(f"|--------|-------|-----------------|")
    if isinstance(cal, dict) and "confidence_bucket" in cal:
        buckets = cal.get("confidence_bucket", [])
        counts = cal.get("count", [])
        accs = cal.get("actual_accuracy", [])
        for b, cnt, acc in zip(buckets, counts, accs):
            acc_str = f"{acc:.1%}" if acc is not None else "N/A"
            md.append(f"| {b} | {cnt} | {acc_str} |")
    md.append("")

    if small_sample_warnings:
        md.append("### Calibration Sample-Size Warnings\n")
        for w in small_sample_warnings:
            md.append(f"- {w}")
        md.append("")

    # --- Multi-run recall range (if available) ---
    mr = all_metrics.get("multi_run_recall", {})
    if mr and mr.get("recall_values"):
        md.append("## Recall Across Repeated Runs\n")
        md.append(f"Recall is reported as a **range** rather than a single point ")
        md.append(f"figure because the Gemini API does not produce deterministic ")
        md.append(f"output despite `temperature=0.0` / `seed=42`.\n")
        md.append(f"| Metric | Value |")
        md.append(f"|--------|-------|")
        md.append(f"| Runs | {mr['num_runs']} |")
        md.append(f"| Cache-only mode | {'Yes' if mr.get('cache_only_mode') else 'No'} |")
        md.append(f"| Recall (single-run snapshot) | {c.get('recall', 'N/A')} |")
        md.append(f"| Recall range (min-max) | {mr['recall_min']}-{mr['recall_max']} |")
        md.append(f"| Recall mean | {mr['recall_mean']} |")
        md.append(f"| TP per run | {mr['tp_values']} |")
        md.append(f"| FN per run | {mr['fn_values']} |")
        if mr.get("skipped_per_run"):
            md.append(f"| Skipped (no cache) per run | {mr['skipped_per_run']} |")
        md.append(f"\n> **Note:** {mr['note']}\n")
        if mr.get("cache_only_mode"):
            md.append("> **Cache-only caveat:** The recall range above is identical across all runs "
                      "because cache was preserved (--cache-only mode). This reflects cached-result "
                      "consistency, not true run-to-run variance. To measure actual API "
                      "non-determinism, run without --cache-only.\n")

    with open("results/evaluation_report.md", "w") as f:
        f.write("\n".join(md))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sentinel — Full pipeline evaluation on frozen test set"
    )
    parser.add_argument(
        "--cache-only", action="store_true", default=False,
        help="Skip ambiguous transactions without cached AI results instead "
             "of calling the API. Makes evaluate.py demo-safe: completes in "
             "seconds using only cached data. Report clearly states how many "
             "transactions were skipped."
    )
    args = parser.parse_args()
    cache_only = args.cache_only

    if cache_only:
        print("CACHE-ONLY MODE: Will skip uncached transactions (no API calls).",
              file=sys.stderr)

    test_df = pd.read_csv("data/transactions_test.csv")
    rel_path = "data/account_relationships.csv"
    if os.path.exists(rel_path):
        relationships_df = pd.read_csv(rel_path)
    else:
        print(
            "WARNING: account_relationships.csv not found. "
            "Ring detection metrics will show 0.",
            file=sys.stderr,
        )
        relationships_df = pd.DataFrame(columns=["account_id_a", "account_id_b", "shared_signal"])

    results_df = run_full_pipeline(test_df, cache_only=cache_only)

    # Report skipped transactions
    skipped = results_df.attrs.get("skipped_no_cache", [])
    if skipped:
        print(f"\nSKIPPED {len(skipped)} transactions (no cached AI result): "
              f"{', '.join(skipped)}", file=sys.stderr)
        print("These are recorded as 'manual_review' in the final decision.",
              file=sys.stderr)

    # --- Debug: recovery eligibility ---
    total_declined = (results_df["status"] == "declined").sum()
    eligible = results_df[
        (results_df["status"] == "declined") & (results_df["final_decision"] != "fraud")
    ]
    print(f"RECOVERY DEBUG: total_declined={total_declined}, "
          f"recovery_eligible={len(eligible)}", file=sys.stderr)
    if len(eligible) > 0:
        print(f"RECOVERY DEBUG: decline_type counts:\n"
              f"{eligible['decline_type'].value_counts().to_string()}\n"
              f"RECOVERY DEBUG: amount stats: min={eligible['amount'].min():.2f}, "
              f"max={eligible['amount'].max():.2f}, "
              f"mean={eligible['amount'].mean():.2f}", file=sys.stderr)
    else:
        print("RECOVERY DEBUG: No recovery-eligible transactions "
              "(all declined txns may have final_decision='fraud')",
              file=sys.stderr)

    metrics = {
        "dataset": "transactions_test.csv",
        "dataset_size": len(test_df),
        "cache_only_mode": cache_only,
        "skipped_no_cache_count": len(skipped),
        "skipped_no_cache_txns": skipped,
        "dataset_note": (
            "Frozen held-out set. Single-run recall is a snapshot; true recall "
            "varies due to Gemini API non-determinism — see multi_run_recall."
        ),
        "classification": compute_classification_metrics(results_df),
        "fraud_typology": compute_fraud_typology(results_df),
        "calibration": compute_calibration_table(results_df).to_dict(orient="list"),
        "ring_detection": compute_ring_accuracy(results_df, relationships_df),
        "recovery": compute_recovery_metrics(results_df),
    }

    # Multi-run recall range
    if cache_only:
        # In cache-only mode, don't clear cache between runs
        print("\nRunning multi-run recall analysis (5 runs, cache-only)...",
              file=sys.stderr)
    else:
        print("\nRunning multi-run recall analysis (5 runs)...", file=sys.stderr)

    multi_run = compute_multi_run_recall(n_runs=5, test_df=test_df,
                                         cache_only=cache_only)
    metrics["multi_run_recall"] = multi_run
    print(f"Recall range: {multi_run['recall_min']}–{multi_run['recall_max']} "
          f"(mean {multi_run['recall_mean']})", file=sys.stderr)

    write_reports(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
