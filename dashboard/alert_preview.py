"""
Alert Preview Mockup — visual only, no real Slack/email integration.

Renders a realistic preview of what a production Slack alert or email
notification would look like for a high-risk transaction. Uses only
Streamlit native components (no HTML, no external dependencies).

This is a MOCKUP for buildathon presentation purposes — it demonstrates
"production thinking" without requiring real infrastructure.
"""

import streamlit as st

from engine.masking import tokenize_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity_emoji(band: str, confidence: int | None = None) -> str:
    """Return a single emoji summarising severity for the alert header."""
    if band == "high_risk":
        if confidence is not None and confidence >= 90:
            return "\U0001f534"  # red circle
        return "\U0001f7e0"      # orange circle
    return "\U0001f7e1"          # yellow circle (ambiguous)


def _top_factor(rule_breakdown: dict) -> str | None:
    """Return the name of the highest-scoring rule, if any."""
    if not rule_breakdown:
        return None
    scores = rule_breakdown if "rule_scores" not in rule_breakdown else rule_breakdown["rule_scores"]
    if not scores:
        return None
    top = max(scores, key=scores.get)
    return top.replace("_", " ").title()


def _masked_txn_id(txn_id: str) -> str:
    """Mask a transaction ID for display in the alert."""
    return tokenize_id(str(txn_id), prefix="TXN")


# ---------------------------------------------------------------------------
# Slack-style alert preview
# ---------------------------------------------------------------------------

def render_slack_alert_preview(
    row: dict,
    ai_result: dict | None = None,
    cf_result: dict | None = None,
) -> None:
    """Render a Slack message-card preview for a flagged transaction.

    Parameters
    ----------
    row : dict
        A single transaction row (Series converted to dict).  Must contain
        at least: txn_id, amount, band, raw_risk_score, rule_breakdown.
    ai_result : dict, optional
        Cached AI investigation result (verdict, confidence, reasoning,
        recommended_action).  If None, AI fields are omitted.
    cf_result : dict, optional
        Counterfactual result dict.  If available, the top contributing
        factor is shown in the alert.
    """
    band = row.get("band", "ambiguous")
    confidence = ai_result.get("confidence") if ai_result else None
    emoji = _severity_emoji(band, confidence)
    top_factor = _top_factor(
        row["rule_breakdown"]
        if isinstance(row["rule_breakdown"], dict)
        else {}
    )
    masked_id = _masked_txn_id(row["txn_id"])
    amount = row.get("amount", 0)
    risk_score = row.get("raw_risk_score", "N/A")

    # --- Slack-style card using native Streamlit ---
    with st.container(border=True):
        # Bot header
        st.markdown(
            f"**:robot_face: Sentinel Fraud Alerts**  \n"
            f"*App  •  Today at 12:00 AM*"
        )
        st.divider()

        # Title
        st.markdown(f"### {emoji} High-Risk Transaction Flagged")

        # Key details as metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Amount", f"INR {amount:,.2f}")
        m2.metric("Risk Score", f"{risk_score}/100")
        m3.metric("Verdict", band.replace("_", " ").title())

        # Detail rows
        st.markdown(f"**Transaction:** `{masked_id}`")
        st.markdown(f"**Account:** `{tokenize_id(str(row.get('account_id', '')), prefix='ACCT')}`")

        if top_factor:
            st.markdown(f"**Top contributing factor:** {top_factor}")

        if ai_result:
            st.markdown(f"**AI Verdict:** {ai_result.get('verdict', 'N/A')} "
                        f"(confidence {confidence}/100)")
            reasoning = ai_result.get("reasoning", "")
            if reasoning:
                # Show first 150 chars of reasoning
                short = reasoning[:150] + ("..." if len(reasoning) > 150 else "")
                st.caption(f"Reasoning: {short}")
            action = ai_result.get("recommended_action", "")
            if action:
                st.markdown(f"**Recommended action:** {action}")
        else:
            st.caption("AI investigation not available for this transaction.")

        if cf_result and cf_result.get("would_flip_verdict"):
            factor = cf_result.get("factor_name", "unknown")
            st.warning(f"Counterfactual: removing **{factor}** would flip the verdict.")

        st.divider()

        # Mock action buttons (non-functional)
        b1, b2, b3 = st.columns(3)
        b1.button("View Case File", key=f"slack_case_{row['txn_id']}", disabled=True)
        b2.button("Approve", key=f"slack_approve_{row['txn_id']}", disabled=True)
        b3.button("Block", key=f"slack_block_{row['txn_id']}", disabled=True)

        st.caption("This is a mockup preview — not a real Slack integration.")


# ---------------------------------------------------------------------------
# Email-style alert preview
# ---------------------------------------------------------------------------

def render_email_alert_preview(
    row: dict,
    ai_result: dict | None = None,
    cf_result: dict | None = None,
) -> None:
    """Render an email notification preview for a flagged transaction.

    Parameters
    ----------
    row : dict
        A single transaction row (Series converted to dict).
    ai_result : dict, optional
        Cached AI investigation result.
    cf_result : dict, optional
        Counterfactual result dict.
    """
    band = row.get("band", "ambiguous")
    confidence = ai_result.get("confidence") if ai_result else None
    emoji = _severity_emoji(band, confidence)
    top_factor = _top_factor(
        row["rule_breakdown"]
        if isinstance(row["rule_breakdown"], dict)
        else {}
    )
    masked_id = _masked_txn_id(row["txn_id"])
    amount = row.get("amount", 0)
    risk_score = row.get("raw_risk_score", "N/A")

    with st.container(border=True):
        # Email header
        st.markdown(
            f"**From:** Sentinel Fraud Monitor "
            f"<sentinel-alerts@yourcompany.com>  \n"
            f"**To:** fraud-ops@yourcompany.com  \n"
            f"**Subject:** {emoji} [{band.upper().replace('_', ' ')}] "
            f"Transaction `{masked_id}` — INR {amount:,.2f}"
        )
        st.divider()

        # Body
        st.markdown(f"### {emoji} High-Risk Transaction Alert")

        st.markdown(
            f"A transaction has been flagged by the Sentinel fraud detection "
            f"pipeline with a risk score of **{risk_score}/100** and classified "
            f"as **{band.replace('_', ' ').upper()}**."
        )

        # Table of details
        st.markdown(
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| Transaction ID | `{masked_id}` |\n"
            f"| Account | `{tokenize_id(str(row.get('account_id', '')), prefix='ACCT')}` |\n"
            f"| Amount | INR {amount:,.2f} |\n"
            f"| Risk Score | {risk_score}/100 |\n"
            f"| Band | {band.replace('_', ' ').title()} |"
        )

        if top_factor:
            st.markdown(f"**Primary risk signal:** {top_factor}")

        if ai_result:
            st.markdown("---")
            st.markdown("**AI Investigation Summary**")
            st.markdown(
                f"- **Verdict:** {ai_result.get('verdict', 'N/A')} "
                f"(confidence {confidence}/100)  \n"
                f"- **Recommended action:** {ai_result.get('recommended_action', 'N/A')}  \n"
                f"- **Reasoning:** {ai_result.get('reasoning', 'N/A')[:200]}"
            )

        if cf_result and cf_result.get("would_flip_verdict"):
            factor = cf_result.get("factor_name", "unknown")
            st.info(
                f"Counterfactual note: removing the **{factor}** signal would "
                f"have flipped this verdict to legitimate."
            )

        st.divider()
        st.caption(
            "[View in Sentinel Dashboard](http://localhost:8501)  •  "
            "[Open Case File](http://localhost:8501)"
        )
        st.caption("This is a mockup preview — not a real email integration.")
