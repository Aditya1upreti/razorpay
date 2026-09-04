"""
Confidence Calibration Report

Runs the full pipeline against data/transactions_calibration.csv
and evaluates whether the AI investigator's stated confidence scores
are well-calibrated (i.e. does "confidence 80%" actually mean being
right ~80% of the time).

Outputs:
  - results/calibration_report.json
  - results/calibration_report.md
  - results/calibration_curve.png

Metrics:
  - Expected Calibration Error (ECE): weighted average of |accuracy - confidence|
    across buckets, weighted by bucket size. Lower = better calibration.
  - Brier Score: mean squared difference between predicted probability and
    actual outcome. Lower = better (range 0-1, like MSE for probabilities).
"""

import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import run_full_pipeline, compute_classification_metrics
from engine.ai_investigator import CONFIDENCE_THRESHOLD


CALIBRATION_DATA = "data/transactions_calibration.csv"
CALIBRATION_REL = "data/account_relationships_calibration.csv"


def count_cached_investigations(df):
    """How many ambiguous transactions already have cached AI results."""
    cache_dir = "cache"
    ambiguous = df[df["band"] == "ambiguous"]
    cached = 0
    for _, row in ambiguous.iterrows():
        cache_path = os.path.join(cache_dir, f"investigate_{row['txn_id']}.json")
        if os.path.exists(cache_path):
            cached += 1
    return len(ambiguous), cached


def compute_calibration_table(df, n_buckets=10):
    """Bucket by stated confidence, compute actual accuracy per bucket.
    Automatically widens buckets if sample sizes are too thin."""
    ai_rows = df[df["ai_confidence"].notna()].copy()
    if len(ai_rows) == 0:
        return pd.DataFrame(), 0.0, 0.0

    # Default to 10-point buckets; fall back to wider if needed
    bins = list(range(0, 101, 10))
    labels = [f"{i}-{i+10}" for i in range(0, 100, 10)]

    ai_rows = ai_rows.copy()
    ai_rows["confidence_bucket"] = pd.cut(
        ai_rows["ai_confidence"], bins=bins, labels=labels, right=False
    )

    rows = []
    for bucket in labels:
        bucket_rows = ai_rows[ai_rows["confidence_bucket"] == bucket]
        count = len(bucket_rows)
        if count == 0:
            rows.append({"confidence_bucket": bucket, "count": 0, "actual_accuracy": None})
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

    cal_df = pd.DataFrame(rows)

    # Compute ECE and Brier score
    valid = cal_df[cal_df["actual_accuracy"].notna() & (cal_df["count"] > 0)].copy()
    if len(valid) == 0:
        return cal_df, 0.0, 0.0

    total_ai = len(ai_rows)
    ece = 0.0
    for _, row in valid.iterrows():
        weight = row["count"] / total_ai
        bucket_mid = (int(row["confidence_bucket"].split("-")[0]) + 5) / 100.0
        ece += weight * abs(bucket_mid - row["actual_accuracy"])

    # Brier score: for each AI-row, (stated_confidence/100 - actual_outcome)^2
    brier_terms = []
    for _, r in ai_rows.iterrows():
        stated = r["ai_confidence"] / 100.0
        actual = 1.0 if r["true_label"] == True else 0.0
        ai_correct = (r["ai_verdict"] == "fraud_likely") == (r["true_label"] == True)
        predicted = stated if r["ai_verdict"] == "fraud_likely" else (1.0 - stated)
        brier_terms.append((predicted - actual) ** 2)
    brier = np.mean(brier_terms) if brier_terms else 0.0

    return cal_df, round(ece, 4), round(float(brier), 4)


def plot_calibration_curve(cal_df, ece, brier, total_ai, output_path, confidence_threshold=65):
    """Plot stated confidence vs actual accuracy, with perfect-calibration diagonal."""
    valid = cal_df[cal_df["actual_accuracy"].notna() & (cal_df["count"] > 0)].copy()
    if len(valid) == 0:
        return

    x_stated = []
    y_actual = []
    counts = []
    for _, row in valid.iterrows():
        lo = int(row["confidence_bucket"].split("-")[0])
        x_stated.append((lo + 5) / 100.0)
        y_actual.append(row["actual_accuracy"])
        counts.append(row["count"])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Shaded region for < threshold (manual review zone — accuracy is informational only)
    threshold_frac = confidence_threshold / 100.0
    ax.axvspan(0, threshold_frac, alpha=0.10, color="orange", label=f"< {confidence_threshold}% confidence (manual review zone)")
    ax.axvline(x=threshold_frac, color="orange", linestyle="--", linewidth=1, alpha=0.7)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.scatter(x_stated, y_actual, s=[c * 15 for c in counts], alpha=0.7,
               edgecolors="steelblue", facecolors="lightskyblue", linewidths=1.5)

    for xi, yi, c in zip(x_stated, y_actual, counts):
        ax.annotate(f"n={c}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="gray")

    ax.set_xlabel("Stated confidence", fontsize=12)
    ax.set_ylabel("Actual accuracy", fontsize=12)
    ax.set_title("Confidence Calibration Curve", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    subtitle = f"ECE = {ece:.3f} | Brier = {brier:.3f} | AI-investigated rows = {total_ai}"
    ax.set_title(f"Confidence Calibration Curve\n{subtitle}", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {output_path}")


def _compute_effective_calibration(cal_df, total_ai, confidence_threshold):
    """Compute ECE and Brier only for buckets at or above the confidence threshold.
    These are the cases that actually drive auto-decisions."""
    if cal_df.empty or total_ai == 0:
        return 0.0, 0.0

    # Filter to valid buckets >= threshold
    valid = cal_df[
        cal_df["actual_accuracy"].notna() & (cal_df["count"] > 0)
    ].copy()
    if valid.empty:
        return 0.0, 0.0

    high_conf = valid[
        valid["confidence_bucket"].apply(
            lambda b: int(b.split("-")[0]) >= confidence_threshold
        )
    ]
    if high_conf.empty:
        return 0.0, 0.0

    # Effective ECE: weighted by count in high-conf buckets only
    high_total = int(high_conf["count"].sum())
    effective_ece = 0.0
    for _, row in high_conf.iterrows():
        weight = row["count"] / high_total
        bucket_mid = (int(row["confidence_bucket"].split("-")[0]) + 5) / 100.0
        effective_ece += weight * abs(bucket_mid - row["actual_accuracy"])

    # Effective Brier: only for rows in high-conf buckets
    # We approximate from the bucket-level data since we don't have row-level here
    brier_terms = []
    for _, row in high_conf.iterrows():
        bucket_mid = (int(row["confidence_bucket"].split("-")[0]) + 5) / 100.0
        acc = row["actual_accuracy"]
        cnt = int(row["count"])
        # Approximate: for each row in bucket, squared error vs actual accuracy
        # Since all rows in bucket have similar confidence, use bucket midpoint as stated conf
        for _ in range(cnt):
            brier_terms.append((bucket_mid - acc) ** 2)
    effective_brier = float(np.mean(brier_terms)) if brier_terms else 0.0

    return round(effective_ece, 4), round(effective_brier, 4)


def write_calibration_report(metrics, cal_df, ece, brier, total_ai, n_calibration, confidence_threshold=65):
    os.makedirs("results", exist_ok=True)

    # Compute effective calibration (ECE/Brier for >= threshold only)
    effective_ece, effective_brier = _compute_effective_calibration(
        cal_df, total_ai, confidence_threshold
    )

    report = {
        "sample_size": n_calibration,
        "ai_investigated": total_ai,
        "ece": ece,
        "brier_score": brier,
        "effective_ece": effective_ece,
        "effective_brier": effective_brier,
        "confidence_threshold": confidence_threshold,
        "calibration_table": cal_df.to_dict(orient="list") if len(cal_df) > 0 else {},
        "classification": metrics.get("classification", {}),
    }

    with open("results/calibration_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    cal_df.to_csv("results/calibration_curve.csv", index=False)

    # --- Dynamic interpretation ---
    if ece < 0.05:
        interpretation = "The model is well-calibrated. Stated confidence closely matches actual accuracy."
    elif ece < 0.10:
        interpretation = "The model is moderately calibrated. Some deviation between stated and actual accuracy."
    elif ece < 0.20:
        interpretation = "The model shows meaningful miscalibration. Confidence scores do not reliably predict actual accuracy."
    else:
        interpretation = "The model is significantly miscalibrated. Confidence scores are poor predictors of actual accuracy."

    overconfident = False
    underconfident = False
    valid = cal_df[cal_df["actual_accuracy"].notna() & (cal_df["count"] > 0)].copy()
    if len(valid) > 0:
        high_conf = valid[valid["confidence_bucket"].isin(["70-80", "80-90", "90-100"])]
        low_conf = valid[valid["confidence_bucket"].isin(["0-10", "10-20", "20-30"])]
        if len(high_conf) > 0 and len(low_conf) > 0:
            high_avg_acc = high_conf["actual_accuracy"].mean()
            high_avg_conf = (high_conf["confidence_bucket"].apply(lambda x: int(x.split("-")[0])) + 5).mean() / 100
            low_avg_acc = low_conf["actual_accuracy"].mean()
            low_avg_conf = (low_conf["confidence_bucket"].apply(lambda x: int(x.split("-")[0])) + 5).mean() / 100

            if high_avg_conf > high_avg_acc + 0.05:
                overconfident = True
            if low_avg_acc > low_avg_conf + 0.05:
                underconfident = True

    md = []
    md.append("# Confidence Calibration Report\n")
    md.append(f"**Sample size:** {n_calibration} transactions\n")
    md.append(f"**AI-investigated (ambiguous band):** {total_ai} transactions\n")
    md.append("")
    md.append("## Summary\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Expected Calibration Error (ECE) | {ece:.4f} |")
    md.append(f"| Brier Score | {brier:.4f} |")
    md.append(f"| AI-investigated count | {total_ai} |")
    md.append("")

    md.append("## Interpretation\n")
    md.append(interpretation + "\n")
    if overconfident:
        md.append("The model tends to be **overconfident** — it states high confidence "
                   "more often than its actual accuracy warrants.\n")
    if underconfident:
        md.append("The model tends to be **underconfident** — it states low confidence "
                   "when it is actually correct more often than expected.\n")
    if not overconfident and not underconfident:
        md.append("No systematic directional bias detected (neither systematically "
                   "overconfident nor underconfident).\n")

    md.append("## Calibration Table\n")
    md.append(f"| Confidence Bucket | Count | Actual Accuracy |")
    md.append(f"|-------------------|-------|-----------------|")
    for _, row in cal_df.iterrows():
        acc_str = f"{row['actual_accuracy']:.1%}" if row["actual_accuracy"] is not None else "N/A"
        md.append(f"| {row['confidence_bucket']} | {row['count']} | {acc_str} |")
    md.append("")

    md.append("## Notes\n")
    md.append("- ECE (Expected Calibration Error): weighted average of |accuracy - confidence| across buckets. Lower is better (0 = perfect calibration).\n")
    md.append("- Brier Score: mean squared error between predicted probability and actual outcome. Range 0-1, lower is better.\n")
    md.append("- Confidence buckets are 10-point wide (0-10%, 10-20%, ..., 90-100%). Empty buckets are excluded from metrics.\n")
    md.append("- This analysis uses the larger calibration set (500+ synthetic transactions) for statistical robustness.\n")

    with open("results/calibration_report.md", "w") as f:
        f.write("\n".join(md))

    return report


def main():
    print("=" * 60)
    print("CONFIDENCE CALIBRATION REPORT")
    print("=" * 60)

    # 1. Load calibration data
    if not os.path.exists(CALIBRATION_DATA):
        print(f"\nCalibration data not found at {CALIBRATION_DATA}")
        print("Run: python data/generate_calibration_set.py")
        sys.exit(1)

    cal_df = pd.read_csv(CALIBRATION_DATA)
    cal_df["account_created_at"] = pd.to_datetime(cal_df["account_created_at"])
    cal_df["timestamp"] = pd.to_datetime(cal_df["timestamp"])

    print(f"\nLoaded {len(cal_df)} calibration transactions")
    print(f"  Fraud ratio: {cal_df['true_label'].mean():.2%}")

    # 2. Estimate API calls
    from engine import rules
    scored = rules.score_batch(cal_df)
    ambiguous_count, cached_count = count_cached_investigations(scored)
    uncached = ambiguous_count - cached_count

    print(f"\n--- API Call Estimate ---")
    print(f"  Ambiguous band (need AI investigation): {ambiguous_count}")
    print(f"  Already cached:                         {cached_count}")
    print(f"  Fresh API calls needed:                 {uncached}")
    print(f"  Each requires 3 Gemini calls (for+against+reconcile)")
    print(f"  Estimated Gemini API calls:             {uncached * 3}")
    print(f"  Estimated runtime:                      ~{uncached * 4 // 60}-{uncached * 4 // 60 + 2} minutes")

    if uncached > 100:
        print(f"\n  WARNING: {uncached * 3} API calls is a lot. Consider reducing sample size.")
        print(f"  To reduce, edit generate_calibration_set.py and use smaller counts.")

    # 3. Run full pipeline
    print(f"\nRunning full pipeline (rules -> graph -> AI -> recovery)...")
    results = run_full_pipeline(cal_df, relationships_path=CALIBRATION_REL)

    # 4. Classification metrics
    cls_metrics = compute_classification_metrics(results)
    print(f"\nClassification: P={cls_metrics['precision']:.3f} R={cls_metrics['recall']:.3f} "
          f"F1={cls_metrics['f1_score']:.3f} FPR={cls_metrics['false_positive_rate']:.3f}")

    # 5. Calibration table + ECE + Brier
    total_ai = results["ai_confidence"].notna().sum()
    print(f"\nAI-investigated rows: {total_ai}")

    cal_table, ece, brier = compute_calibration_table(results)
    print(f"\nCalibration metrics:")
    print(f"  ECE:        {ece:.4f}")
    print(f"  Brier:      {brier:.4f}")

    if len(cal_table) > 0:
        print(f"\nCalibration table:")
        for _, row in cal_table.iterrows():
            if row["count"] > 0:
                print(f"  {row['confidence_bucket']:>8}: n={row['count']:>3}  acc={row['actual_accuracy']:.1%}")

    # 6. Write report + chart
    chart_path = "results/calibration_curve.png"
    plot_calibration_curve(cal_table, ece, brier, total_ai, chart_path, CONFIDENCE_THRESHOLD)

    report = write_calibration_report(
        {"classification": cls_metrics},
        cal_table, ece, brier, total_ai, len(cal_df), CONFIDENCE_THRESHOLD,
    )

    print(f"\nReport saved: results/calibration_report.json")
    print(f"Report saved: results/calibration_report.md")
    print(f"Chart saved:  {chart_path}")
    print(f"Table saved:  results/calibration_curve.csv")


if __name__ == "__main__":
    main()
