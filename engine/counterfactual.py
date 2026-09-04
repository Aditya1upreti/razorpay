"""
Counterfactual Explanation Layer (read-only, post-hoc analysis)

For any transaction that was auto-blocked, flagged by the AI review, or sent
to manual review, this layer answers: "what single input change would have
flipped the verdict?"

How it works
------------
Deterministic factor-perturbation. It re-runs the EXISTING Layer 1 rule
functions (rules.velocity_score, amount_pattern_score, account_age_score,
geo_mismatch_score via rules.score_transaction) and Layer 2 ring detection
(graph_builder.find_ring_candidates / explain_ring_bfs), neutralizing ONE
contributing factor at a time, and checks whether the resulting score crosses
a band threshold (30 / 75 by default) that would change the verdict.

Neutralization detail
---------------------
The rule factors read overlapping inputs (card_fingerprint and timestamp feed
multiple scorers), so mutating a single transaction field would leak
side-effects into other factors. Instead we recompute the full breakdown and
zero out the target factor's points — the only perturbation that touches
exactly one factor.

Ring signal note
----------------
In the current pipeline `in_detected_ring` is injected only as context for
the AI review step; it does NOT contribute to raw_risk_score. Removing ring
membership therefore leaves the rule score unchanged, and that is reported
honestly rather than faked.

Honesty guarantee
-----------------
- No LLM / API calls. Explanations are plain f-string templating — free and
  perfectly reproducible (no hallucination surface).
- If NO single factor removal flips the verdict, we say so explicitly
  ("flagged by multiple independent factors") instead of inventing a
  counterfactual — consistent with the project's "insufficient evidence"
  principle in engine/ai_investigator.py.
- This layer never stores or alters the decision that was actually made.
"""

import json
import os
import sys

import pandas as pd

try:
    from . import graph_builder, rules
except ImportError:  # running as a plain script (python engine/counterfactual.py)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import graph_builder, rules

RULE_FACTOR_ORDER = ["velocity", "amount_pattern", "account_age", "geo_mismatch"]

RULE_FACTOR_LABELS = {
    "velocity": "the card-velocity signal (repeated attempts inside a short window)",
    "amount_pattern": "the amount-pattern signal (large charge following small test payments)",
    "account_age": "the account-age signal (account created shortly before this charge)",
    "geo_mismatch": "the geo-mismatch signal (billing region differing from the IP region)",
}


def _as_dict(transaction):
    if hasattr(transaction, "to_dict"):
        return transaction.to_dict()
    return dict(transaction)


def _parse_breakdown(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _band_of(score, low_threshold, high_threshold):
    if score < low_threshold:
        return "safe"
    if score > high_threshold:
        return "high_risk"
    return "ambiguous"


def _would_flip(original_score, counterfactual_score, low_threshold, high_threshold):
    """Does neutralizing this factor move the score across the threshold that
    would change the verdict? (Factors only ever lower the score.)"""
    band = _band_of(original_score, low_threshold, high_threshold)
    if band == "high_risk":            # auto-blocked -> not auto-blocked anymore
        return counterfactual_score <= high_threshold
    if band == "ambiguous":            # escalated -> auto-clear
        return counterfactual_score < low_threshold
    return False                       # already safe: nothing to flip


def _ring_context(txn, graph):
    account_id = txn.get("account_id")
    if graph is None or not account_id:
        return None
    try:
        candidates = graph_builder.find_ring_candidates(graph)
    except Exception:
        return None
    for ring in candidates:
        if account_id in ring:
            signals = set()
            sub = graph.subgraph(ring)
            for neighbor in sub.neighbors(account_id):
                sig = sub.edges[account_id, neighbor].get("signal", "shared attribute")
                signals.add(str(sig))
            trail = ""
            try:
                trail = graph_builder.explain_ring_bfs(graph, ring)
            except Exception:
                pass
            return {
                "ring_size": len(ring),
                "signals": sorted(signals),
                "trail": trail,
            }
    return None


def _current_action(band, final_decision, score, low, high):
    if band == "high_risk" or score > high:
        return "auto-blocked as fraud"
    if score < low:
        return "auto-approved with zero review"
    if final_decision in ("manual_review", "insufficient_evidence"):
        return "sent to manual review"
    if final_decision in ("fraud", "flagged", "blocked", "fraud_likely"):
        return "flagged as a fraud candidate"
    if final_decision in ("cleared", "legitimate", "legitimate_likely"):
        return "cleared as legitimate"
    return "escalated for AI review"


def _flip_action(band, counterfactual_score, low, high):
    if band == "high_risk":
        if counterfactual_score < low:
            return "auto-approved with zero review"
        return "escalated for review"
    if band == "ambiguous":
        return "auto-approved with zero review"
    return "escalated for review"


def _rule_factor_explanation(name, points, original_score, cf_score, band,
                             final_decision, would_flip, low, high):
    label = RULE_FACTOR_LABELS[name]
    if would_flip:
        flip_action = _flip_action(band, cf_score, low, high)
        cur_action = _current_action(band, final_decision, original_score, low, high)
        return (
            f"If {label} were absent (instead of contributing {points} of the "
            f"{original_score} risk points), the raw_risk_score would have been "
            f"{cf_score} instead of {original_score}, and this case would have "
            f"{flip_action} instead of {cur_action}."
        )
    return (
        f"Removing {label} alone (worth {points} of the {original_score} risk "
        f"points) would only lower the score to {cf_score} — not enough to "
        f"cross the threshold required to clear this case."
    )


def _ring_explanation(ring, original_score):
    size = ring.get("ring_size")
    signals = ring.get("signals", [])
    size_txt = f"{size} linked accounts" if size else "a linked account cluster"
    sig_txt = f" (shared signals: {', '.join(signals)})" if signals else ""
    return (
        f"This account is part of a detected ring — {size_txt}{sig_txt}. Ring "
        f"membership is injected only as context for the AI review step and does "
        f"not change the rule score, so removing it would leave raw_risk_score at "
        f"{original_score} (unchanged). No deterministic single-input change would "
        f"flip this verdict."
    )


def _no_flip_explanation(original_score, breakdown, ring, band, low, high):
    contributing = [f"{k} (+{v})" for k, v in breakdown.items() if int(v) > 0]
    if band == "high_risk":
        band_txt = "high-risk (auto-block)"
    elif band == "ambiguous":
        band_txt = "ambiguous (escalation)"
    else:
        band_txt = "safe (auto-clear)"

    if contributing:
        joined = ", ".join(contributing)
        ring_note = " and membership in a detected account ring" if ring else ""
        return (
            f"This case is flagged by multiple independent factors{ring_note} — no "
            f"single change would clear it (score {original_score}, {band_txt} band). "
            f"Contributing rule factors: {joined}. Only removing several at once "
            f"would cross the required threshold."
        )
    return (
        f"This transaction carries no rule-level flags (raw_risk_score "
        f"{original_score}) — the verdict rests on signals outside the "
        f"deterministic rule engine, so no deterministic single-input change "
        f"would flip it."
    )


def generate_counterfactual(transaction, current_result=None, all_transactions=None,
                            graph=None, low_threshold=30, high_threshold=75):
    """
    Compute a single-factor counterfactual explanation for one transaction.

    Args:
        transaction: dict or pandas Series with the rule-engine input fields
            (card_fingerprint, timestamp, amount, account_created_at,
            billing_region, ip_region, account_id).
        current_result: optional dict with pipeline context:
            txn_id, raw_risk_score, rule_breakdown, band, in_detected_ring,
            final_decision ("fraud" / "legitimate" / "manual_review").
        all_transactions: DataFrame of the batch this transaction was scored
            against (needed by velocity/amount-pattern scorers). If None, the
            transaction is scored in isolation.
        graph: networkx graph with ring membership (from graph_builder).
        low_threshold / high_threshold: band boundaries (defaults 30 / 75).

    Returns:
        dict:
            txn_id, band, original_score, current_action,
            factor_name (most decisive flip, or None),
            counterfactual_score, would_flip_verdict,
            explanation_text (plain-string, deterministic),
            factors (per-factor breakdown),
            in_detected_ring.
    """
    txn = _as_dict(transaction)
    current_result = current_result or {}
    final_decision = str(current_result.get("final_decision", ""))

    if all_transactions is None:
        all_transactions = pd.DataFrame([txn])
        for col in ("timestamp", "account_created_at"):
            if col in all_transactions.columns:
                all_transactions[col] = pd.to_datetime(all_transactions[col], errors="coerce")

    baseline = rules.score_transaction(txn, all_transactions)
    original_score = int(baseline["raw_risk_score"])
    breakdown = dict(baseline["rule_breakdown"])

    band = _band_of(original_score, low_threshold, high_threshold)
    txn_id = txn.get("txn_id", current_result.get("txn_id", "?"))

    factors = []

    for name in RULE_FACTOR_ORDER:
        points = int(breakdown.get(name, 0))
        if points <= 0:
            continue
        other_total = sum(p for k, p in breakdown.items() if k != name)
        cf_score = min(other_total, 100)
        would_flip = _would_flip(original_score, cf_score, low_threshold, high_threshold)
        factors.append({
            "factor_name": name,
            "factor_label": RULE_FACTOR_LABELS[name],
            "original_points": points,
            "original_score": original_score,
            "counterfactual_score": cf_score,
            "would_flip_verdict": would_flip,
            "explanation_text": _rule_factor_explanation(
                name, points, original_score, cf_score, band,
                final_decision, would_flip, low_threshold, high_threshold,
            ),
        })

    ring = _ring_context(txn, graph)
    if ring is not None:
        factors.append({
            "factor_name": "ring",
            "factor_label": "shared-attribute ring membership",
            "original_points": None,
            "original_score": original_score,
            "counterfactual_score": original_score,
            "would_flip_verdict": False,
            "explanation_text": _ring_explanation(ring, original_score),
        })

    flippers = [f for f in factors if f["would_flip_verdict"]]
    if flippers:
        best = max(flippers, key=lambda f: f["original_points"] or 0)
        chosen_name = best["factor_name"]
        cf_score = best["counterfactual_score"]
        explanation = best["explanation_text"]
    else:
        chosen_name = None
        cf_score = original_score
        explanation = _no_flip_explanation(
            original_score, breakdown, ring, band, low_threshold, high_threshold,
        )

    return {
        "txn_id": txn_id,
        "band": band,
        "original_score": original_score,
        "current_action": _current_action(
            band, final_decision, original_score, low_threshold, high_threshold,
        ),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "factors": factors,
        "factor_name": chosen_name,
        "counterfactual_score": cf_score,
        "would_flip_verdict": bool(chosen_name is not None),
        "explanation_text": explanation,
        "in_detected_ring": ring is not None,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/transactions_test.csv")
    df["account_created_at"] = pd.to_datetime(df["account_created_at"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    scored = rules.score_batch(df)
    graph = graph_builder.build_graph(scored)
    all_ring_accounts = set()
    for r in graph_builder.find_ring_candidates(graph):
        all_ring_accounts |= r
    scored["in_detected_ring"] = scored["account_id"].isin(all_ring_accounts)

    for band_name in ("high_risk", "ambiguous"):
        subset = scored[scored["band"] == band_name]
        if subset.empty:
            print(f"No {band_name} rows.\n")
            continue
        row = subset.iloc[0]
        context = {
            "txn_id": row["txn_id"],
            "band": row["band"],
            "raw_risk_score": row["raw_risk_score"],
            "rule_breakdown": row["rule_breakdown"],
            "in_detected_ring": bool(row["in_detected_ring"]),
            "final_decision": "manual_review" if band_name == "ambiguous" else "fraud",
        }
        cf = generate_counterfactual(row, context, all_transactions=scored, graph=graph)
        print(f"=== {row['txn_id']} ({row['band']}, score={row['raw_risk_score']}) ===")
        print(f"  flips_verdict={cf['would_flip_verdict']}  "
              f"flip_factor={cf['factor_name']}  cf_score={cf['counterfactual_score']}")
        print(f"  {cf['explanation_text']}")
        print(f"  in_detected_ring={cf['in_detected_ring']}")
        for f in cf["factors"]:
            print(f"    - {f['factor_name']:16} pts={f['original_points']}  "
                  f"-> score if removed={f['counterfactual_score']}  flip={f['would_flip_verdict']}")
        print()