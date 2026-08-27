"""
DAY 5 — Track 03 Fork: Revenue Recovery

Takes transactions that were NOT flagged as fraud but still failed
(status = declined). Routes based on decline_type:

  soft_decline -> silent automated retry (no customer contact
                  simulated), capped attempts, stopping rule
  hard_decline -> simulated customer-facing outreach, capped +
                  suppression-list-aware, capped attempts

Tracks: recovered ₹ vs at-risk ₹, split by category.
"""

import pandas as pd

MAX_SILENT_RETRIES = 2
MAX_OUTREACH_ATTEMPTS = 2

# Mapping from specific decline_type values to soft/hard categories
# Based on dataset generation logic
SOFT_DECLINE_TYPES = {"timeout", "insufficient_funds"}
HARD_DECLINE_TYPES = {"stolen_card", "expired_card"}


def route_decline(transaction) -> str:
    """Return 'silent_retry' or 'customer_outreach' based on
    transaction['decline_type']."""
    decline_type = transaction["decline_type"]

    if decline_type in SOFT_DECLINE_TYPES:
        return "silent_retry"
    elif decline_type in HARD_DECLINE_TYPES:
        return "customer_outreach"
    else:
        raise ValueError(
            f"Unexpected decline_type: {decline_type!r}. "
            f"Expected one of {SOFT_DECLINE_TYPES | HARD_DECLINE_TYPES}."
        )


def simulate_silent_retry(transaction, max_attempts=MAX_SILENT_RETRIES) -> dict:
    """Simulate retrying a soft-declined payment. Since this is
    synthetic data, YOU decide the success probability/outcome in
    a controlled, explainable way (not random for demo credibility —
    e.g. tie outcome to a field in your synthetic data).
    Returns: {attempts_made, recovered: bool, recovered_amount}"""
    amount = transaction["amount"]
    txn_id = transaction["txn_id"]

    attempts_made = 0
    recovered = False

    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        if amount < 2000:
            # Small amounts succeed on attempt 1
            recovered = True
            break
        elif 2000 <= amount <= 6000:
            # Mid-range amounts need a retry, succeed on attempt 2
            if attempt >= 2:
                recovered = True
                break
        else:
            # Amount > 6000: too large to silently recover, fail all
            pass

    recovered_amount = amount if recovered else 0.0

    return {
        "txn_id": txn_id,
        "attempts_made": attempts_made,
        "recovered": recovered,
        "recovered_amount": recovered_amount,
    }


def simulate_customer_outreach(transaction, max_attempts=MAX_OUTREACH_ATTEMPTS,
                                suppression_list=None) -> dict:
    """Simulate outreach for a hard-declined payment. Must respect
    a suppression list (accounts that opted out) — check it before
    attempting.
    Returns: {attempts_made, recovered: bool, recovered_amount, suppressed: bool}"""
    txn_id = transaction["txn_id"]
    account_id = transaction["account_id"]
    amount = transaction["amount"]

    # Check suppression list first
    if suppression_list is not None and account_id in suppression_list:
        return {
            "txn_id": txn_id,
            "attempts_made": 0,
            "recovered": False,
            "recovered_amount": 0.0,
            "suppressed": True,
        }

    attempts_made = 0
    recovered = False

    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        if amount < 3000:
            # Small amounts: likely to respond, succeed on attempt 1
            recovered = True
            break
        elif 3000 <= amount <= 4500:
            # Mid-range: succeed on attempt 2
            if attempt >= 2:
                recovered = True
                break
        else:
            # Amount > 4500: too large, fail all attempts
            pass

    recovered_amount = amount if recovered else 0.0

    return {
        "txn_id": txn_id,
        "attempts_made": attempts_made,
        "recovered": recovered,
        "recovered_amount": recovered_amount,
        "suppressed": False,
    }


def run_recovery_batch(declined_transactions_df, suppression_list=None) -> dict:
    """Run recovery across a batch. Returns summary:
    {
      total_at_risk: float,
      total_recovered: float,
      soft_decline_recovered: float,
      hard_decline_recovered: float,
      soft_decline_count: int,
      hard_decline_count: int,
      suppressed_count: int,
      per_transaction_log: [...]   # for the audit trail / dashboard
    }"""
    results = []
    per_transaction_log = []

    total_at_risk = 0.0
    total_recovered = 0.0
    soft_decline_recovered = 0.0
    hard_decline_recovered = 0.0
    soft_decline_count = 0
    hard_decline_count = 0
    suppressed_count = 0

    for _, row in declined_transactions_df.iterrows():
        transaction = row.to_dict()
        route = route_decline(transaction)

        if route == "silent_retry":
            soft_decline_count += 1
            result = simulate_silent_retry(transaction)
            result["route"] = route
            total_at_risk += transaction["amount"]
            if result["recovered"]:
                total_recovered += result["recovered_amount"]
                soft_decline_recovered += result["recovered_amount"]
        elif route == "customer_outreach":
            hard_decline_count += 1
            result = simulate_customer_outreach(transaction, suppression_list=suppression_list)
            result["route"] = route
            total_at_risk += transaction["amount"]
            if result.get("suppressed", False):
                suppressed_count += 1
            elif result["recovered"]:
                total_recovered += result["recovered_amount"]
                hard_decline_recovered += result["recovered_amount"]

        per_transaction_log.append(result)

    return {
        "total_at_risk": total_at_risk,
        "total_recovered": total_recovered,
        "soft_decline_recovered": soft_decline_recovered,
        "hard_decline_recovered": hard_decline_recovered,
        "soft_decline_count": soft_decline_count,
        "hard_decline_count": hard_decline_count,
        "suppressed_count": suppressed_count,
        "per_transaction_log": per_transaction_log,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/transactions_train.csv")
    declined = df[df["status"] == "declined"].copy()

    # Sanity check: are any declined transactions also fraud?
    fraud_declined = declined[declined["true_label"] == True]
    if len(fraud_declined) > 0:
        print(f"WARNING: {len(fraud_declined)} declined transactions are also fraud!")
        print(fraud_declined[["txn_id", "decline_type", "amount", "true_label"]])
    else:
        print("Sanity check passed: no declined transactions are flagged as fraud.\n")

    # Create a small fake suppression list (1-2 account_ids from declined set)
    # Use hard_decline accounts to test the suppression path
    hard_decline_accounts = declined[declined["decline_type"].isin(HARD_DECLINE_TYPES)]["account_id"].unique()
    suppression_list = list(hard_decline_accounts[:2]) if len(hard_decline_accounts) > 0 else []
    print(f"Suppression list (test): {suppression_list}\n")

    # Run recovery batch
    summary = run_recovery_batch(declined, suppression_list=suppression_list)

    # Print summary
    print("=" * 60)
    print("RECOVERY BATCH SUMMARY")
    print("=" * 60)
    print(f"  Total at risk:          INR {summary['total_at_risk']:,.2f}")
    print(f"  Total recovered:        INR {summary['total_recovered']:,.2f}")
    if summary["total_at_risk"] > 0:
        recovery_pct = (summary["total_recovered"] / summary["total_at_risk"]) * 100
    else:
        recovery_pct = 0.0
    print(f"  Recovery rate:          {recovery_pct:.1f}%")
    print()
    print(f"  Soft decline count:     {summary['soft_decline_count']}")
    print(f"  Soft decline recovered: INR {summary['soft_decline_recovered']:,.2f}")
    print(f"  Hard decline count:     {summary['hard_decline_count']}")
    print(f"  Hard decline recovered: INR {summary['hard_decline_recovered']:,.2f}")
    print(f"  Suppressed count:       {summary['suppressed_count']}")
    print()
    print("PER-TRANSACTION LOG:")
    print("-" * 60)
    for entry in summary["per_transaction_log"]:
        print(f"  {entry['txn_id']} | {entry['route']:20s} | "
              f"attempts={entry['attempts_made']} | "
              f"recovered={entry['recovered']} | "
              f"amount=INR {entry['recovered_amount']:,.2f}"
              + (f" | suppressed={entry['suppressed']}" if "suppressed" in entry else ""))
