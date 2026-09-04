"""
Case Export — Structured case files for fraud analysts

Generates a single exportable document per transaction containing
everything a human fraud analyst would need to review or defend
a decision: rule scores, graph/ring trail, AI investigator arguments,
final verdict, confidence, and counterfactual analysis.

Read-only — does not modify or re-run any part of the decision pipeline.
All customer-identifying fields are masked via engine/masking.py.
"""

import json
import os
import sys
from datetime import datetime, timezone

from .masking import build_masked_context, tokenize_id

try:
    from . import graph_builder, counterfactual
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import graph_builder, counterfactual


def generate_case_file(
    transaction_id: str,
    scored_row,
    cached_investigation: dict | None = None,
    all_transactions=None,
    graph=None,
    ring_candidates: list | None = None,
) -> dict:
    """Assemble a structured case file for one transaction.

    Args:
        transaction_id: the txn_id to build a case for.
        scored_row: pandas Series or dict with scored transaction data
            (must include raw_risk_score, rule_breakdown, band,
            in_detected_ring, and the original transaction fields).
        cached_investigation: dict from load_cached_investigation() or None.
        all_transactions: full scored DataFrame (needed for counterfactual).
        graph: networkx graph (needed for ring trail and counterfactual).
        ring_candidates: list of ring account sets (needed for ring trail).

    Returns:
        dict with the full case file structure.
    """
    row = scored_row.to_dict() if hasattr(scored_row, "to_dict") else dict(scored_row)

    txn_id = row.get("txn_id", transaction_id)
    breakdown = row.get("rule_breakdown")
    if isinstance(breakdown, str):
        breakdown = json.loads(breakdown)
    breakdown = breakdown or {}

    raw_risk_score = row.get("raw_risk_score", 0)
    band = row.get("band", "unknown")
    in_detected_ring = bool(row.get("in_detected_ring", False))

    # --- Transaction summary (masked) ---
    rule_info = {"rule_scores": breakdown, "raw_risk_score": raw_risk_score}
    masked = build_masked_context(row, rule_info)

    transaction_summary = {
        "txn_token": masked.get("txn_token", tokenize_id(txn_id, prefix="TXN")),
        "account_token": masked.get("account_token"),
        "card_token": masked.get("card_token"),
        "amount": row.get("amount"),
        "timestamp": str(row.get("timestamp", "")),
        "status": row.get("status"),
        "billing_region": row.get("billing_region"),
        "ip_region": row.get("ip_region"),
        "account_age_minutes": masked.get("account_age_minutes"),
    }

    # --- Layer 1: Rule scores ---
    layer1 = {
        "raw_risk_score": raw_risk_score,
        "band": band,
        "factors": {
            "velocity": breakdown.get("velocity", 0),
            "amount_pattern": breakdown.get("amount_pattern", 0),
            "account_age": breakdown.get("account_age", 0),
            "geo_mismatch": breakdown.get("geo_mismatch", 0),
        },
    }

    # --- Layer 2: Graph / ring signal ---
    layer2 = None
    if in_detected_ring and graph is not None and ring_candidates:
        ring_accounts = _find_ring_for_account(txn_id, row.get("account_id"), ring_candidates)
        if ring_accounts:
            trail = ""
            try:
                trail = graph_builder.explain_ring_bfs(graph, ring_accounts)
            except Exception:
                trail = "(BFS trail unavailable)"

            connected = []
            for acct in ring_accounts:
                if acct != row.get("account_id"):
                    connected.append({
                        "account_token": tokenize_id(str(acct), prefix="ACCT"),
                        "shared_signal": _get_shared_signal(graph, row.get("account_id"), acct),
                    })

            layer2 = {
                "in_detected_ring": True,
                "ring_size": len(ring_accounts),
                "connected_accounts": connected,
                "bfs_trail": trail,
            }
    else:
        layer2 = {
            "in_detected_ring": False,
        }

    # --- Layer 3: AI investigation ---
    layer3 = None
    if cached_investigation:
        confidence = cached_investigation.get("confidence")
        threshold = 65
        override_triggered = (
            confidence is not None
            and confidence < threshold
            and cached_investigation.get("verdict") != "insufficient_evidence"
        )

        layer3 = {
            "argument_for_fraud": cached_investigation.get("argument_for"),
            "argument_against_fraud": cached_investigation.get("argument_against"),
            "reconciled_verdict": cached_investigation.get("verdict", "unknown"),
            "confidence": confidence,
            "confidence_threshold": threshold,
            "threshold_override_triggered": override_triggered,
            "degraded_mode": cached_investigation.get("degraded_mode", False),
            "reasoning": cached_investigation.get("reasoning"),
            "recommended_action": cached_investigation.get("recommended_action"),
        }
    else:
        layer3 = {
            "argument_for_fraud": None,
            "argument_against_fraud": None,
            "reconciled_verdict": "not_investigated",
            "confidence": None,
            "confidence_threshold": 65,
            "threshold_override_triggered": False,
            "degraded_mode": False,
            "reasoning": "No cached AI investigation available for this transaction.",
            "recommended_action": "manual_review",
        }

    # --- Final decision ---
    final_decision = _infer_final_decision(band, cached_investigation)
    decision_ts = datetime.now(timezone.utc).isoformat()

    # --- Counterfactual analysis ---
    counterfactual_section = None
    if all_transactions is not None:
        try:
            cf = counterfactual.generate_counterfactual(
                row,
                current_result={
                    "txn_id": txn_id,
                    "band": band,
                    "raw_risk_score": raw_risk_score,
                    "rule_breakdown": json.dumps(breakdown),
                    "in_detected_ring": in_detected_ring,
                    "final_decision": final_decision,
                },
                all_transactions=all_transactions,
                graph=graph,
            )
            counterfactual_section = {
                "would_flip_verdict": cf.get("would_flip_verdict", False),
                "most_influential_factor": cf.get("factor_name"),
                "counterfactual_score": cf.get("counterfactual_score"),
                "explanation_text": cf.get("explanation_text"),
                "factors": cf.get("factors", []),
            }
        except Exception:
            counterfactual_section = {"error": "Counterfactual analysis unavailable."}

    # --- Assemble case file ---
    case_file = {
        "case_id": txn_id,
        "generated_at": decision_ts,
        "transaction_summary": transaction_summary,
        "layer1_rule_scores": layer1,
        "layer2_graph_signal": layer2,
        "layer3_ai_investigation": layer3,
        "final_decision": {
            "verdict": final_decision,
            "decision_timestamp": decision_ts,
        },
        "counterfactual_analysis": counterfactual_section,
    }

    return case_file


def _find_ring_for_account(txn_id, account_id, ring_candidates):
    """Find the ring set containing this account, if any."""
    if not account_id or not ring_candidates:
        return None
    for ring in ring_candidates:
        if account_id in ring:
            return ring
    return None


def _get_shared_signal(graph, acct_a, acct_b):
    """Get the edge signal between two accounts, if connected."""
    if graph is None or not acct_a or not acct_b:
        return None
    try:
        return graph.edges[acct_a, acct_b].get("signal", "shared attribute")
    except (KeyError, Exception):
        return None


def _infer_final_decision(band, cached):
    """Replicate pipeline final-decision logic."""
    if band == "high_risk":
        return "fraud"
    if band == "safe":
        return "legitimate"
    if cached:
        verdict = cached.get("verdict")
        confidence = cached.get("confidence")
        if verdict == "fraud_likely" and confidence is not None and confidence >= 65:
            return "fraud"
        if verdict == "legitimate_likely" and confidence is not None and confidence >= 65:
            return "legitimate"
    return "manual_review"


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_case_json(case_file: dict, output_path: str):
    """Write the case file as pretty-printed JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(case_file, f, indent=2, ensure_ascii=False, default=str)
    return output_path


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def export_case_pdf(case_file: dict, output_path: str):
    """Write the case file as a clean, readable one-page PDF."""
    from fpdf import FPDF

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Title ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _sanitize(f"Sentinel Case File: {case_file.get('case_id', '?')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, _sanitize(f"Generated: {case_file.get('generated_at', '?')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Transaction Summary ---
    _section_header(pdf, "Transaction Summary")
    ts = case_file.get("transaction_summary", {})
    for key, val in ts.items():
        _field_row(pdf, key.replace("_", " ").title(), str(val) if val is not None else "N/A")

    # --- Layer 1: Rule Scores ---
    _section_header(pdf, "Layer 1: Rule Engine Scores")
    l1 = case_file.get("layer1_rule_scores", {})
    _field_row(pdf, "Raw Risk Score", f"{l1.get('raw_risk_score', '?')} / 100")
    _field_row(pdf, "Band", l1.get("band", "?"))
    factors = l1.get("factors", {})
    for fname, pts in factors.items():
        _field_row(pdf, f"  {fname.replace('_', ' ').title()}", f"{pts} pts")

    # --- Layer 2: Graph / Ring ---
    _section_header(pdf, "Layer 2: Graph / Ring Signal")
    l2 = case_file.get("layer2_graph_signal", {})
    if l2 and l2.get("in_detected_ring"):
        _field_row(pdf, "In Detected Ring", "Yes")
        _field_row(pdf, "Ring Size", f"{l2.get('ring_size', '?')} accounts")
        connected = l2.get("connected_accounts", [])
        if connected:
            _field_row(pdf, "Connected Accounts", "")
            for conn in connected:
                _field_row(pdf, f"  {conn.get('account_token', '?')}", f"via {conn.get('shared_signal', '?')}")
        trail = l2.get("bfs_trail", "")
        if trail:
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 5, "BFS Traversal Trail:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Courier", "", 8)
            for line in trail.split("\n"):
                pdf.cell(0, 4, _sanitize(f"  {line}"), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _sanitize("No ring signal for this transaction."), new_x="LMARGIN", new_y="NEXT")

    # --- Layer 3: AI Investigation ---
    _section_header(pdf, "Layer 3: AI Investigator")
    l3 = case_file.get("layer3_ai_investigation", {})
    if l3:
        _field_row(pdf, "Verdict", l3.get("reconciled_verdict", "?"))
        conf = l3.get("confidence")
        _field_row(pdf, "Confidence", f"{conf} / 100" if conf is not None else "N/A")
        _field_row(pdf, "Threshold", f"{l3.get('confidence_threshold', 65)}")
        override = l3.get("threshold_override_triggered", False)
        _field_row(pdf, "Threshold Override Triggered", "Yes" if override else "No")
        _field_row(pdf, "Degraded Mode", "Yes" if l3.get("degraded_mode") else "No")
        _field_row(pdf, "Reasoning", str(l3.get("reasoning", "N/A"))[:120])
        _field_row(pdf, "Recommended Action", l3.get("recommended_action", "N/A"))

        arg_for = l3.get("argument_for_fraud")
        arg_against = l3.get("argument_against_fraud")
        if arg_for or arg_against:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, _sanitize("Adversarial Arguments:"), new_x="LMARGIN", new_y="NEXT")
            if arg_for:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(0, 5, _sanitize("  FOR fraud:"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 8)
                _pdf_multi_cell(pdf, arg_for, max_width=170)
            if arg_against:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(0, 5, _sanitize("  AGAINST fraud:"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 8)
                _pdf_multi_cell(pdf, arg_against, max_width=170)
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _sanitize("No AI investigation data available."), new_x="LMARGIN", new_y="NEXT")

    # --- Final Decision ---
    _section_header(pdf, "Final Decision")
    fd = case_file.get("final_decision", {})
    verdict = fd.get("verdict", "?")
    pdf.set_font("Helvetica", "B", 11)
    color = {"fraud": (220, 50, 50), "manual_review": (200, 150, 0), "legitimate": (50, 160, 50)}
    rgb = color.get(verdict, (0, 0, 0))
    pdf.set_text_color(*rgb)
    pdf.cell(0, 8, _sanitize(f"Verdict: {verdict.upper()}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    _field_row(pdf, "Decision Timestamp", fd.get("decision_timestamp", "?"))

    # --- Counterfactual Analysis ---
    cf = case_file.get("counterfactual_analysis")
    if cf and "error" not in cf:
        _section_header(pdf, "Counterfactual Analysis")
        flips = cf.get("would_flip_verdict", False)
        _field_row(pdf, "Would Single Factor Flip Verdict", "Yes" if flips else "No")
        if cf.get("most_influential_factor"):
            _field_row(pdf, "Most Influential Factor", cf["most_influential_factor"])
        if cf.get("counterfactual_score") is not None:
            _field_row(pdf, "Counterfactual Score", str(cf["counterfactual_score"]))
        explanation = cf.get("explanation_text", "")
        if explanation:
            pdf.set_font("Helvetica", "I", 9)
            _pdf_multi_cell(pdf, explanation, max_width=170)

    pdf.output(output_path)
    return output_path


def _sanitize(text):
    """Replace unicode characters unsupported by latin-1 core fonts."""
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u2192": "->",   # right arrow
        "\u2190": "<-",   # left arrow
        "\u2022": "*",    # bullet
        "\u00a0": " ",    # nbsp
        "\u00b0": " deg", # degree
        "\u2248": "~",    # approximately
        "\u2264": "<=",   # less than or equal
        "\u2265": ">=",   # greater than or equal
    }
    s = str(text)
    for old, new in replacements.items():
        s = s.replace(old, new)
    # Fallback: replace any remaining non-latin-1 chars
    result = []
    for ch in s:
        try:
            ch.encode("latin-1")
            result.append(ch)
        except UnicodeEncodeError:
            result.append("?")
    return "".join(result)


def _section_header(pdf, title):
    """Render a section header with a thin line."""
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 7, _sanitize(f"  {title}"), new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(1)


def _field_row(pdf, label, value):
    """Render a label: value row."""
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(60, 5, _sanitize(f"{label}:"), new_x="END")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, _sanitize(str(value)), new_x="LMARGIN", new_y="NEXT")


def _pdf_multi_cell(pdf, text, max_width=170):
    """Render multi-line text, truncating if too long for one page."""
    pdf.set_font("Helvetica", "", 8)
    lines = _sanitize(text).split("\n")
    for line in lines:
        if len(line) > 200:
            line = line[:197] + "..."
        pdf.multi_cell(max_width, 4, f"  {line}")
