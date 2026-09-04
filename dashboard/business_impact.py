"""
Business Impact — Translates pipeline metrics into merchant-facing numbers.

Read-only layer: computes nothing new, only reformats existing pipeline
output (evaluation_report.json + recovery results) into plain business terms.
"""


def compute_business_impact(
    scored_df,
    classification_metrics: dict,
    recovery_metrics: dict,
) -> dict:
    """Assemble business-impact numbers from existing pipeline output.

    Args:
        scored_df: DataFrame with columns true_label, band, amount
            (the scored test set with pipeline results).
            If a final_decision column exists it is used directly;
            otherwise it is derived from band (high_risk=fraud,
            safe=legitimate, ambiguous=manual_review).
        classification_metrics: dict from evaluate.compute_classification_metrics().
        recovery_metrics: dict from evaluate.compute_recovery_metrics().

    Returns:
        dict with business-facing numbers and plain-language notes.
    """
    # --- Fraud losses avoided ---
    # Determine final_decision: use column if present, else derive from band
    if "final_decision" in scored_df.columns:
        decision_col = scored_df["final_decision"]
    else:
        decision_col = scored_df["band"].map({
            "high_risk": "fraud",
            "safe": "legitimate",
            "ambiguous": "manual_review",
        })
    fraud_blocked = scored_df[
        (scored_df["true_label"] == True) & (decision_col == "fraud")
    ]
    total_fraud_amount_blocked = round(float(fraud_blocked["amount"].sum()), 2)
    fraud_blocked_count = int(len(fraud_blocked))

    # --- False positives ---
    fp_rate = classification_metrics.get("false_positive_rate", 0.0)
    fp_count = classification_metrics.get("fp", 0)

    # --- Recovery ---
    total_recovered = recovery_metrics.get("total_recovered", 0.0)
    recovery_rate = recovery_metrics.get("recovery_rate_pct", 0.0)
    total_at_risk = recovery_metrics.get("total_at_risk", 0.0)

    # --- Manual review ---
    manual_review_pct = classification_metrics.get("manual_review_pct", 0.0)
    manual_review_count = classification_metrics.get("manual_review_count", 0)
    total_count = classification_metrics.get("total_count", 0)

    return {
        "fraud_amount_blocked": total_fraud_amount_blocked,
        "fraud_blocked_count": fraud_blocked_count,
        "false_positive_rate": fp_rate,
        "false_positive_count": fp_count,
        "total_recovered": total_recovered,
        "recovery_rate_pct": recovery_rate,
        "total_at_risk": total_at_risk,
        "manual_review_pct": manual_review_pct,
        "manual_review_count": manual_review_count,
        "total_transactions": total_count,
    }


def compute_before_after_narrative(
    scored_df,
    classification_metrics: dict,
    recovery_metrics: dict,
) -> dict:
    """Compute a before/after narrative from the held-out test set.

    "Without Sentinel" = what happens if every transaction is auto-approved
    with no fraud detection. "With Sentinel" = what the pipeline actually caught
    and recovered.

    All numbers are derived from the same scored_df and existing metrics —
    no invented assumptions.

    Returns:
        dict with without_sentinel, with_sentinel, and caveat fields.
    """
    # --- Without Sentinel: all fraud goes through, no recovery ---
    all_fraud = scored_df[scored_df["true_label"] == True]
    total_fraud_count = int(len(all_fraud))
    total_fraud_amount = round(float(all_fraud["amount"].sum()), 2)

    # --- With Sentinel: what was caught and recovered ---
    # Determine final_decision: use column if present, else derive from band
    if "final_decision" in scored_df.columns:
        decision_col = scored_df["final_decision"]
    else:
        decision_col = scored_df["band"].map({
            "high_risk": "fraud",
            "safe": "legitimate",
            "ambiguous": "manual_review",
        })
    fraud_caught = scored_df[
        (scored_df["true_label"] == True) & (decision_col == "fraud")
    ]
    caught_count = int(len(fraud_caught))
    caught_amount = round(float(fraud_caught["amount"].sum()), 2)

    missed_count = total_fraud_count - caught_count
    missed_amount = round(float(
        all_fraud[~all_fraud.index.isin(fraud_caught.index)]["amount"].sum()
    ), 2)

    false_positive_count = classification_metrics.get("fp", 0)
    total_recovered = recovery_metrics.get("total_recovered", 0.0)
    total_at_risk = recovery_metrics.get("total_at_risk", 0.0)

    recall = classification_metrics.get("recall", 0.0)

    return {
        "without_sentinel": {
            "fraud_cases": total_fraud_count,
            "fraud_amount": total_fraud_amount,
            "recovered": 0.0,
        },
        "with_sentinel": {
            "fraud_caught": caught_count,
            "fraud_amount_blocked": caught_amount,
            "fraud_missed": missed_count,
            "fraud_amount_missed": missed_amount,
            "false_positives": false_positive_count,
            "recovered": total_recovered,
            "recovery_at_risk": total_at_risk,
            "recall_pct": round(recall * 100, 1),
        },
        "caveat": (
            f"Based on a {classification_metrics.get('total_count', 96)}-transaction "
            f"held-out test set. Even with Sentinel, {missed_count} fraud case(s) "
            f"were not caught (see Business Impact tab for full technical metrics)."
        ),
    }
