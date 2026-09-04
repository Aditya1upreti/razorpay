"""
Threshold simulation for risk-tolerance what-if analysis.

Pure computation — no API calls, no I/O. Takes a scored DataFrame and
new threshold values, returns band distribution and classification stats
for the proposed configuration.

NOTE: This simulates band changes only. It does NOT re-run AI investigation
for ambiguous cases — it shows what WOULD be classified under new thresholds.
The AI investigation results are only available for the original run.
"""

import pandas as pd


def simulate_thresholds(df: pd.DataFrame, low_threshold: int = 30, high_threshold: int = 75) -> dict:
    """
    Simulate classification metrics under proposed threshold values.

    Args:
        df: DataFrame with columns 'raw_risk_score' and 'true_label'
        low_threshold: lower bound for safe band (score < low -> safe)
        high_threshold: upper bound for high_risk band (score > high -> high_risk)

    Returns:
        dict with precision, recall, f1_score, false_positive_rate, fpr_pct,
             tp, fp, tn, fn, safe_count, ambiguous_count, high_risk_count,
             low_threshold, high_threshold
    """
    if "raw_risk_score" not in df.columns:
        raise ValueError("DataFrame must have 'raw_risk_score' column")
    if "true_label" not in df.columns:
        raise ValueError("DataFrame must have 'true_label' column")

    # Classify each transaction
    scores = df["raw_risk_score"]
    labels = df["true_label"]

    # Classification: safe (auto-clear), ambiguous (manual review), high_risk (auto-flag)
    predicted_safe = scores < low_threshold
    predicted_high_risk = scores > high_threshold
    predicted_ambiguous = ~(predicted_safe | predicted_high_risk)

    # Count by band
    safe_count = int(predicted_safe.sum())
    ambiguous_count = int(predicted_ambiguous.sum())
    high_risk_count = int(predicted_high_risk.sum())

    # For binary classification (fraud vs legitimate), we only evaluate
    # the auto-clear and auto-flag decisions. Ambiguous cases are unknown
    # until manual review, so we exclude them from precision/recall.
    evaluated = predicted_safe | predicted_high_risk
    tp = int(((predicted_high_risk) & labels).sum())
    fp = int(((predicted_high_risk) & (~labels)).sum())
    tn = int(((predicted_safe) & (~labels)).sum())
    fn = int(((predicted_safe) & labels).sum())

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "fpr_pct": round(false_positive_rate * 100, 2),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "safe_count": safe_count,
        "ambiguous_count": ambiguous_count,
        "high_risk_count": high_risk_count,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }


def compute_threshold_deltas(baseline: dict, simulated: dict) -> dict:
    """
    Compute the delta between baseline and simulated metrics.

    Args:
        baseline: dict from simulate_thresholds with baseline thresholds
        simulated: dict from simulate_thresholds with proposed thresholds

    Returns:
        dict with delta values for key metrics
    """
    return {
        "precision_delta": round(simulated["precision"] - baseline["precision"], 4),
        "recall_delta": round(simulated["recall"] - baseline["recall"], 4),
        "f1_score_delta": round(simulated["f1_score"] - baseline["f1_score"], 4),
        "fpr_pct_delta": round(simulated["fpr_pct"] - baseline["fpr_pct"], 4),
        "safe_count_delta": simulated["safe_count"] - baseline["safe_count"],
        "ambiguous_count_delta": simulated["ambiguous_count"] - baseline["ambiguous_count"],
        "high_risk_count_delta": simulated["high_risk_count"] - baseline["high_risk_count"],
    }


def find_optimal_thresholds(
    df: pd.DataFrame,
    fp_cost: float,
    fn_cost: float,
    low_range: tuple = (10, 60),
    high_range: tuple = (40, 90),
    step: int = 1,
) -> dict:
    """Find threshold pair that minimises total expected cost.

    Performs a grid search over low_threshold/high_threshold combinations,
    calling simulate_thresholds() for each. Total cost is:
        (false_positive_count * fp_cost) + (false_negative_count * fn_cost)

    Args:
        df: scored DataFrame with 'raw_risk_score' and 'true_label' columns.
        fp_cost: user-supplied assumed cost of one false positive (INR).
        fn_cost: user-supplied assumed cost of one missed fraud (INR).
        low_range: (min, max) for lower threshold search.
        high_range: (min, max) for upper threshold search.
        step: grid step size (1 = every point, 2 = every other, etc.).

    Returns:
        dict with:
            optimal_low, optimal_high: the best threshold pair
            optimal_metrics: simulate_thresholds() output for the winning pair
            total_cost: minimum total expected cost
            baseline_cost: total cost at default (30, 75)
            baseline_metrics: simulate_thresholds() output at (30, 75)
            fp_cost, fn_cost: the input cost assumptions
            grid_size: number of combinations evaluated
    """
    best_cost = float("inf")
    best_result = None

    low_values = range(low_range[0], low_range[1] + 1, step)
    high_values = range(high_range[0], high_range[1] + 1, step)

    grid_size = 0
    for low in low_values:
        for high in high_values:
            if low >= high:
                continue
            result = simulate_thresholds(df, low, high)
            cost = result["fp"] * fp_cost + result["fn"] * fn_cost
            grid_size += 1
            if cost < best_cost:
                best_cost = cost
                best_result = result

    # Baseline at default thresholds
    baseline = simulate_thresholds(df, 30, 75)
    baseline_cost = baseline["fp"] * fp_cost + baseline["fn"] * fn_cost

    return {
        "optimal_low": best_result["low_threshold"],
        "optimal_high": best_result["high_threshold"],
        "optimal_metrics": best_result,
        "total_cost": best_cost,
        "baseline_cost": baseline_cost,
        "baseline_metrics": baseline,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "grid_size": grid_size,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    # Quick sanity check with train data
    df = pd.read_csv("data/transactions_test.csv")
    from engine.rules import score_batch
    scored = score_batch(df)

    # Baseline thresholds (30, 75)
    baseline = simulate_thresholds(scored, 30, 75)
    print("=== Baseline (30, 75) ===")
    for k, v in baseline.items():
        print(f"  {k}: {v}")

    # Simulate tighter thresholds (40, 60)
    simulated = simulate_thresholds(scored, 40, 60)
    print("\n=== Simulated (40, 60) ===")
    for k, v in simulated.items():
        print(f"  {k}: {v}")

    deltas = compute_threshold_deltas(baseline, simulated)
    print("\n=== Deltas ===")
    for k, v in deltas.items():
        print(f"  {k}: {v}")
