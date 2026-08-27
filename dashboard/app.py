"""
DAY 7 — Streamlit Dashboard (offline-only, uses cached AI results)

Run with: streamlit run dashboard/app.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from engine import rules, graph_builder, recovery
from engine.masking import build_masked_context

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "cache")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sentinel — AI Risk & Recovery", layout="wide")
st.title("🛡 Sentinel — AI Risk Manager + Revenue Recovery")

# ---------------------------------------------------------------------------
# Data loading (cached via st.cache_data so it only runs once)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions_train.csv"))
    df["account_created_at"] = pd.to_datetime(df["account_created_at"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_scored_data():
    df = load_data()
    df = rules.score_batch(df)
    return df


@st.cache_data
def load_graph_data():
    df = load_data()
    graph = graph_builder.build_graph(df)
    rings = graph_builder.find_ring_candidates(graph)
    return graph, rings


def load_cached_investigation(txn_id: str) -> dict | None:
    """Return cached AI result for txn_id, or None if not cached."""
    path = os.path.join(CACHE_DIR, f"investigate_{txn_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Pre-load everything
# ---------------------------------------------------------------------------
scored_df = load_scored_data()
graph, ring_candidates = load_graph_data()

# Identify ring accounts
all_ring_accounts = set()
for ring in ring_candidates:
    all_ring_accounts |= ring
scored_df["in_detected_ring"] = scored_df["account_id"].isin(all_ring_accounts)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_summary, tab_fraud, tab_ring, tab_ambiguous, tab_recovery = st.tabs(
    ["Summary", "Fraud Case", "Ring Detection", "Ambiguous Case", "Recovery"]
)

# ============================= SUMMARY TAB ================================
with tab_summary:
    st.header("Batch Summary")

    total = len(scored_df)
    band_counts = scored_df["band"].value_counts()

    safe_count = int(band_counts.get("safe", 0))
    ambiguous_count = int(band_counts.get("ambiguous", 0))
    high_risk_count = int(band_counts.get("high_risk", 0))

    # Check cache status for ambiguous transactions
    ambiguous_txns = scored_df[scored_df["band"] == "ambiguous"]["txn_id"].tolist()
    cached_count = sum(
        1 for tid in ambiguous_txns if load_cached_investigation(tid) is not None
    )
    pending_count = ambiguous_count - cached_count

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", total)
    c2.metric("Safe (auto-clear)", safe_count)
    c3.metric("Ambiguous (AI-reviewed)", ambiguous_count)
    c4.metric("High Risk (auto-flag)", high_risk_count)

    st.divider()

    c5, c6 = st.columns(2)
    c5.metric("Ambiguous — Cached AI Result", cached_count)
    c6.metric("Ambiguous — Pending Investigation", pending_count)

    st.divider()
    st.subheader("Band Distribution")
    band_df = pd.DataFrame({
        "Band": ["safe", "ambiguous", "high_risk"],
        "Count": [safe_count, ambiguous_count, high_risk_count],
    })
    st.bar_chart(band_df.set_index("Band"))

    # Fraud ground truth breakdown
    fraud_count = int(scored_df["true_label"].sum())
    legit_count = total - fraud_count
    st.divider()
    st.subheader("Ground Truth (for reference)")
    c7, c8 = st.columns(2)
    c7.metric("Legitimate Transactions", legit_count)
    c8.metric("Fraud Transactions", fraud_count)

# ============================= FRAUD CASE TAB ============================
with tab_fraud:
    st.header("Flagged Transaction — Evidence & Debate")

    high_risk_df = scored_df[scored_df["band"] == "high_risk"].copy()

    if high_risk_df.empty:
        st.warning("No high-risk transactions found in this dataset.")
    else:
        selected_txn = st.selectbox(
            "Select a high-risk transaction",
            high_risk_df["txn_id"].tolist(),
            format_func=lambda x: f"{x} — INR {high_risk_df.loc[high_risk_df['txn_id']==x, 'amount'].values[0]:,.2f}",
        )

        if selected_txn:
            row = high_risk_df[high_risk_df["txn_id"] == selected_txn].iloc[0]

            # Rule breakdown
            st.subheader("Rule Breakdown")
            breakdown = json.loads(row["rule_breakdown"]) if isinstance(row["rule_breakdown"], str) else row["rule_breakdown"]

            rule_cols = st.columns(len(breakdown))
            for i, (rule_name, pts) in enumerate(breakdown.items()):
                rule_cols[i].metric(rule_name.replace("_", " ").title(), f"{pts} pts")

            st.metric("Raw Risk Score", f"{row['raw_risk_score']} / 100")

            # Masked transaction details
            st.subheader("Masked Transaction Details")
            rule_info = {"rule_scores": breakdown, "raw_risk_score": row["raw_risk_score"]}
            masked = build_masked_context(row.to_dict(), rule_info)
            for key, val in masked.items():
                if key == "rule_breakdown":
                    continue
                st.text(f"{key}: {val}")

            # Ground truth
            st.divider()
            label = "FRAUD (true positive)" if row["true_label"] else "LEGITIMATE (false positive)"
            st.info(f"**Ground truth:** {label}")

# ============================= RING DETECTION TAB ========================
with tab_ring:
    st.header("Ring Detection")

    if not ring_candidates:
        st.warning("No ring candidates detected (need clusters of 3+ linked accounts).")
    else:
        st.success(f"Detected **{len(ring_candidates)}** ring candidate(s) involving **{len(all_ring_accounts)}** accounts.")

        for i, ring in enumerate(ring_candidates, 1):
            with st.expander(f"Ring {i} — {len(ring)} accounts", expanded=(i == 1)):
                st.subheader("Accounts Involved")
                for acct in sorted(ring):
                    txn_count = len(scored_df[scored_df["account_id"] == acct])
                    fraud_count = len(scored_df[(scored_df["account_id"] == acct) & (scored_df["true_label"] == True)])
                    st.text(f"  {acct}  ({txn_count} txns, {fraud_count} fraud)")

                st.subheader("BFS Traversal Trail")
                trail = graph_builder.explain_ring_bfs(graph, ring)
                st.code(trail, language=None)

# ============================= AMBIGUOUS CASE TAB ========================
with tab_ambiguous:
    st.header("Ambiguous / Escalated Case")

    ambiguous_df = scored_df[scored_df["band"] == "ambiguous"].copy()

    if ambiguous_df.empty:
        st.warning("No ambiguous transactions found.")
    else:
        # Partition into cached vs pending
        ambiguous_df["has_cache"] = ambiguous_df["txn_id"].apply(
            lambda tid: load_cached_investigation(tid) is not None
        )
        cached_df = ambiguous_df[ambiguous_df["has_cache"]]
        pending_df = ambiguous_df[~ambiguous_df["has_cache"]]

        if pending_df.shape[0] > 0:
            st.warning(
                f"**{pending_df.shape[0]}** ambiguous transaction(s) have **no cached AI result** "
                f"and will not be investigated (API quota exhausted)."
            )
            pending_ids = pending_df["txn_id"].tolist()
            st.multiselect(
                "Pending (not investigated)",
                pending_ids,
                default=pending_ids,
                disabled=True,
                key="pending_txns",
            )
            st.divider()

        if cached_df.empty:
            st.info("No cached AI investigation results available for ambiguous transactions.")
        else:
            cached_options = cached_df["txn_id"].tolist()
            selected_amb = st.selectbox(
                "Select an investigated ambiguous transaction",
                cached_options,
                format_func=lambda x: f"{x} — INR {cached_df.loc[cached_df['txn_id']==x, 'amount'].values[0]:,.2f}",
            )

            if selected_amb:
                row = ambiguous_df[ambiguous_df["txn_id"] == selected_amb].iloc[0]
                cached = load_cached_investigation(selected_amb)

                # Rule breakdown
                st.subheader("Rule Breakdown")
                breakdown = json.loads(row["rule_breakdown"]) if isinstance(row["rule_breakdown"], str) else row["rule_breakdown"]
                rule_cols = st.columns(len(breakdown))
                for i, (rule_name, pts) in enumerate(breakdown.items()):
                    rule_cols[i].metric(rule_name.replace("_", " ").title(), f"{pts} pts")
                st.metric("Raw Risk Score", f"{row['raw_risk_score']} / 100")

                # AI verdict
                st.divider()
                st.subheader("AI Investigator Verdict")

                verdict = cached.get("verdict", "unknown")
                confidence = cached.get("confidence")
                reasoning = cached.get("reasoning", "N/A")
                action = cached.get("recommended_action", "N/A")

                if verdict == "fraud_likely":
                    st.error(f"**Verdict:** {verdict.upper()}")
                elif verdict == "legitimate_likely":
                    st.success(f"**Verdict:** {verdict.upper()}")
                else:
                    st.warning(f"**Verdict:** {verdict.upper()}")

                if confidence is not None:
                    st.metric("Confidence", f"{confidence} / 100")
                    if confidence < 65:
                        st.warning(
                            "⚠️ **Confidence below threshold** — automatically escalated to "
                            "manual review, regardless of the AI's stated verdict."
                        )
                else:
                    st.metric("Confidence", "N/A")

                st.text(f"**Reasoning:** {reasoning}")
                st.text(f"**Recommended Action:** {action}")

                # Adversarial arguments (if present in cache)
                argument_for = cached.get("argument_for")
                argument_against = cached.get("argument_against")

                if argument_for or argument_against:
                    st.divider()
                    st.subheader("Adversarial Debate")
                    col_for, col_again = st.columns(2)
                    with col_for:
                        st.markdown("**🔴 Arguing FOR fraud:**")
                        st.text(argument_for or "Not available for this cached result.")
                    with col_again:
                        st.markdown("**🟢 Arguing AGAINST fraud:**")
                        st.text(argument_against or "Not available for this cached result.")
                else:
                    st.caption("Adversarial argument transcripts not available for this cached result.")

# ============================= RECOVERY TAB ==============================
with tab_recovery:
    st.header("Revenue Recovery")

    declined_df = scored_df[
        (scored_df["status"] == "declined") & (scored_df["true_label"] == False)
    ].copy()

    if declined_df.empty:
        st.info("No non-fraud declined transactions in this dataset to recover.")
        st.caption("Recovery only applies to declined transactions that are NOT fraud.")
    else:
        recovery_results = recovery.run_recovery_batch(declined_df)

        total_at_risk = recovery_results["total_at_risk"]
        total_recovered = recovery_results["total_recovered"]
        recovery_pct = (total_recovered / total_at_risk * 100) if total_at_risk > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total At Risk", f"INR {total_at_risk:,.2f}")
        c2.metric("Total Recovered", f"INR {total_recovered:,.2f}")
        c3.metric("Recovery Rate", f"{recovery_pct:.1f}%")

        st.divider()
        st.subheader("Breakdown by Decline Type")

        breakdown_data = pd.DataFrame({
            "Route": ["Soft (silent retry)", "Hard (customer outreach)"],
            "At Risk": [
                recovery_results.get("soft_decline_recovered", 0),
                recovery_results.get("hard_decline_recovered", 0),
            ],
            "Recovered": [
                recovery_results.get("soft_decline_recovered", 0),
                recovery_results.get("hard_decline_recovered", 0),
            ],
        })
        st.bar_chart(breakdown_data.set_index("Route"))

        # Per-transaction log
        st.divider()
        st.subheader("Per-Transaction Recovery Log")
        log = recovery_results.get("per_transaction_log", [])
        if log:
            log_df = pd.DataFrame(log)
            st.dataframe(log_df, use_container_width=True)
