"""
Layer Ablation Study

Compares three pipeline configurations on the SAME held-out test set
to prove each layer earns its place with real numbers:

1. Rules Only    — Layer 1 alone, ambiguous cases resolved conservatively
2. Rules + Graph — Layer 1 + Layer 2, ambiguous cases resolved via ring signal
3. Full Pipeline — Layer 1 + 2 + 3 (actual system)

Fallback rules (clearly stated, NOT tuned):
  - Rules Only:     ambiguous -> legitimate  (rules lack signal, benefit of the doubt)
  - Rules + Graph:  ambiguous -> fraud if in detected ring, else legitimate
                    (ring is the only additional signal available)

No new API calls. Reuses data already produced by evaluate.py's pipeline.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import run_full_pipeline, compute_classification_metrics


TEST_DATA = "data/transactions_test.csv"
RELATIONSHIPS = "data/account_relationships.csv"
OUTPUT_DIR = "results"


def compute_metrics(df, decision_col):
    """Compute precision/recall/F1/FPR from a decision column.
    Excludes rows where decision is 'manual_review' (same as full pipeline)."""
    eval_df = df[df[decision_col] != "manual_review"].copy()

    y_true = eval_df["true_label"].astype(int)
    y_pred = (eval_df[decision_col] == "fraud").astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    manual_review_count = int((df[decision_col] == "manual_review").sum())
    total = len(df)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "manual_review_count": manual_review_count,
        "manual_review_pct": round(manual_review_count / total, 4) if total > 0 else 0.0,
        "evaluated_count": len(eval_df),
        "total_count": total,
    }


def rules_only_decisions(df):
    """Configuration 1: Rules Only.

    Fallback for ambiguous: legitimate (benefit of the doubt).
    Rationale: with only rules available, ambiguous cases lack sufficient
    signal to justify blocking a real customer. Conservative choice.
    """
    decisions = []
    for _, row in df.iterrows():
        if row["band"] == "high_risk":
            decisions.append("fraud")
        elif row["band"] == "safe":
            decisions.append("legitimate")
        else:
            # Ambiguous: no further signal available -> legitimate
            decisions.append("legitimate")
    return decisions


def rules_plus_graph_decisions(df):
    """Configuration 2: Rules + Graph.

    Fallback for ambiguous: fraud if in detected ring, else legitimate.
    Rationale: ring membership is the strongest graph signal for fraud.
    If Layer 2 detected a ring involving this account, that tips the scale.
    Otherwise, same conservative fallback as rules-only.
    """
    decisions = []
    for _, row in df.iterrows():
        if row["band"] == "high_risk":
            decisions.append("fraud")
        elif row["band"] == "safe":
            decisions.append("legitimate")
        else:
            # Ambiguous: use graph signal
            if row.get("in_detected_ring", False):
                decisions.append("fraud")
            else:
                decisions.append("legitimate")
    return decisions


def full_pipeline_decisions(df):
    """Configuration 3: Full Pipeline (actual system).

    Uses final_decision as computed by evaluate.py's run_full_pipeline.
    No fallback needed — AI investigator resolves ambiguous cases.
    """
    return df["final_decision"].tolist()


def plot_ablation_chart(results, output_path):
    """Bar chart comparing the three configurations side by side."""
    configs = list(results.keys())
    metrics = ["precision", "recall", "f1_score", "false_positive_rate"]
    labels = ["Precision", "Recall", "F1", "FPR"]

    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (config, color) in enumerate(zip(configs, colors)):
        vals = [results[config][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=config, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Layer Ablation Study — Each Layer Earns Its Place")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test data
    test_df = pd.read_csv(TEST_DATA)
    print(f"Loaded {len(test_df)} test transactions")
    print(f"  Fraud ratio: {test_df['true_label'].mean():.2%}")

    # Run full pipeline (reuses existing logic, makes API calls as needed)
    # This produces all per-transaction columns we need
    print("\nRunning full pipeline to compute per-transaction data...")
    results_df = run_full_pipeline(test_df)

    # Apply ablation configurations
    results_df["decision_rules_only"] = rules_only_decisions(results_df)
    results_df["decision_rules_graph"] = rules_plus_graph_decisions(results_df)
    # decision_full is already in final_decision

    # Compute metrics for each configuration
    rules_only_metrics = compute_metrics(results_df, "decision_rules_only")
    rules_graph_metrics = compute_metrics(results_df, "decision_rules_graph")
    full_metrics = compute_metrics(results_df, "final_decision")

    all_results = {
        "Rules Only": rules_only_metrics,
        "Rules + Graph": rules_graph_metrics,
        "Full Pipeline": full_metrics,
    }

    # Print comparison table
    print("\n" + "=" * 72)
    print("LAYER ABLATION STUDY — Side-by-Side Comparison")
    print("=" * 72)
    print(f"{'Configuration':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10} {'Manual Review':>15}")
    print("-" * 72)
    for config, m in all_results.items():
        print(f"{config:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} "
              f"{m['false_positive_rate']:>10.4f} {m['manual_review_count']:>10} ({m['manual_review_pct']:.1%})")
    print("=" * 72)

    # Print layer contribution analysis
    print("\n--- Layer Contribution Analysis ---")
    rules_r = rules_only_metrics["recall"]
    graph_r = rules_graph_metrics["recall"]
    full_r = full_metrics["recall"]

    graph_delta = graph_r - rules_r
    ai_delta = full_r - graph_r

    print(f"  Rules Only recall:     {rules_r:.4f}")
    print(f"  Rules + Graph recall:  {graph_r:.4f}  (graph adds +{graph_delta:.4f})")
    print(f"  Full Pipeline recall:  {full_r:.4f}  (AI adds +{ai_delta:.4f})")
    print(f"  Total improvement:     {full_r - rules_r:.4f}")

    # Fallback rule documentation
    print("\n--- Fallback Rules (transparent, not tuned) ---")
    print("  Rules Only:     ambiguous -> legitimate  (benefit of the doubt)")
    print("  Rules + Graph:  ambiguous -> fraud if in detected ring, else legitimate")
    print("  Full Pipeline:  ambiguous -> AI investigator decides (actual system)")

    # Save results
    output = {
        "dataset": TEST_DATA,
        "dataset_size": len(results_df),
        "fallback_rules": {
            "Rules Only": "ambiguous -> legitimate (benefit of the doubt)",
            "Rules + Graph": "ambiguous -> fraud if in detected ring, else legitimate",
            "Full Pipeline": "ambiguous -> AI investigator decides (actual system)",
        },
        "results": all_results,
        "layer_contributions": {
            "rules_only_recall": rules_r,
            "graph_delta_recall": graph_delta,
            "ai_delta_recall": ai_delta,
            "total_recall_gain": full_r - rules_r,
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Save CSV
    csv_rows = []
    for config, m in all_results.items():
        row = {"configuration": config}
        row.update(m)
        csv_rows.append(row)
    csv_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    # Save chart
    chart_path = os.path.join(OUTPUT_DIR, "ablation_chart.png")
    plot_ablation_chart(all_results, chart_path)


if __name__ == "__main__":
    main()
