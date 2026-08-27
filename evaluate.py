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


def run_full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run every layer end to end on the given dataframe, return
    df with added columns: raw_risk_score, verdict, confidence,
    recommended_action, degraded_mode, etc."""
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
    relationships_df = pd.read_csv("data/account_relationships.csv")

    # --- Layer 3: AI Investigator (ambiguous band only) ---
    ambiguous_mask = df["band"] == "ambiguous"
    ambiguous_indices = df[ambiguous_mask].index

    ai_verdicts = [None] * len(df)
    ai_confidences = [None] * len(df)
    ai_reasonings = [None] * len(df)
    ai_actions = [None] * len(df)
    degraded_modes = [False] * len(df)

    for i, idx in enumerate(ambiguous_indices):
        row = df.loc[idx]
        rule_breakdown = json.loads(row["rule_breakdown"]) if isinstance(row["rule_breakdown"], str) else row["rule_breakdown"]

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
                    print(f"429 RATE LIMITED on {txn_id}: retryDelay={delay}s (attempt {attempt+1}/{max_attempts})", file=sys.stderr)
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

    with open("results/evaluation_report.md", "w") as f:
        f.write("\n".join(md))


def main():
    test_df = pd.read_csv("data/transactions_test.csv")
    relationships_df = pd.read_csv("data/account_relationships.csv")

    results_df = run_full_pipeline(test_df)

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
        "classification": compute_classification_metrics(results_df),
        "calibration": compute_calibration_table(results_df).to_dict(orient="list"),
        "ring_detection": compute_ring_accuracy(results_df, relationships_df),
        "recovery": compute_recovery_metrics(results_df),
    }

    write_reports(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
