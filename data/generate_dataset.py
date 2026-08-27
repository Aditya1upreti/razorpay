"""
DAY 1 — Synthetic dataset generator

Produces:
  - transactions_train.csv   (70% of data, used for rule tuning)
  - transactions_test.csv    (30% of data, FROZEN after generation)
  - account_relationships.csv (ground truth for ring detection)

Target composition (~300 total transactions):
  - ~250 legitimate transactions
  - ~35  card-testing fraud sequences
  - ~15  ring-abuse fraud (multiple accounts sharing signals)
  - ~15-20 "hard negatives" — legit but suspicious-looking
"""

import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad",
    "Chennai", "Kolkata", "Pune", "Jaipur", "Lucknow",
    "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal",
    "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra",
]

INDIAN_STATES = [
    "Maharashtra", "Delhi", "Karnataka", "Telangana", "Gujarat",
    "Tamil Nadu", "West Bengal", "Rajasthan", "Uttar Pradesh",
    "Madhya Pradesh", "Bihar", "Punjab", "Haryana", "Odisha",
    "Chhattisgarh",
]


def _rand_account_id():
    return "ACCT_" + fake.hexify(text="^^^^^^^^")


def _rand_card_id():
    return "CARD_" + fake.hexify(text="^^^^^^^^")


def _rand_device_id():
    return "DEV_" + fake.hexify(text="^^^^^^^^")


def _rand_txn_id(counter):
    return f"TXN{counter:05d}"


def generate_legitimate_transactions(n=250):
    """Generate normal purchase behavior -- varied amounts, spread over time,
    consistent geo/device per account, mix of approved and declined-but-not-fraud."""
    rows = []
    txn_counter = 1

    num_accounts = random.randint(50, 80)
    accounts = []
    for _ in range(num_accounts):
        acct_id = _rand_account_id()
        device = _rand_device_id()
        region = random.choice(INDIAN_CITIES)
        created_at = fake.date_time_between(start_date="-365d", end_date="-60d")
        num_cards = random.randint(2, 3)
        cards = [_rand_card_id() for _ in range(num_cards)]
        accounts.append({
            "account_id": acct_id,
            "device_id": device,
            "billing_region": region,
            "ip_region": region,
            "account_created_at": created_at,
            "cards": cards,
        })

    txns_per_account, remainder = divmod(n, num_accounts)
    txns_counts = [txns_per_account] * num_accounts
    for i in range(remainder):
        txns_counts[i] += 1

    for acct, count in zip(accounts, txns_counts):
        acct_created = acct["account_created_at"]
        acct_first_txn = acct_created + timedelta(days=random.randint(14, 60))
        cards_remaining = list(acct["cards"])
        random.shuffle(cards_remaining)
        card_usage = {c: 0 for c in cards_remaining}

        for j in range(count):
            txn_id = _rand_txn_id(txn_counter)
            txn_counter += 1
            offset = timedelta(
                days=random.randint(0, 29),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            timestamp = acct_first_txn + offset
            amount = round(random.uniform(50, 8000), 2)

            preferred = [c for c in cards_remaining if card_usage[c] < 2]
            if not preferred:
                preferred = cards_remaining
            card = random.choice(preferred)
            card_usage[card] += 1

            rows.append({
                "txn_id": txn_id,
                "card_fingerprint": card,
                "device_id": acct["device_id"],
                "ip_region": acct["ip_region"],
                "account_id": acct["account_id"],
                "account_created_at": acct_created,
                "amount": amount,
                "timestamp": timestamp,
                "billing_region": acct["billing_region"],
                "status": "approved",
                "decline_type": "",
                "true_label": False,
            })

    df = pd.DataFrame(rows)
    return df


def generate_card_testing_fraud(n=35):
    """4-8 small transactions (R1-20) within a 2-5 min window,
    same card_fingerprint, followed by one large transaction,
    often from a newly-created account."""
    rows = []
    txn_counter = 9000

    num_sequences = random.randint(6, 7)
    total_small = n - num_sequences
    small_per_seq, remainder = divmod(total_small, num_sequences)
    seq_sizes = [small_per_seq] * num_sequences
    for i in range(remainder):
        seq_sizes[i] += 1

    base_date = fake.date_time_between(start_date="-20d", end_date="-3d")

    for seq_idx, small_count in enumerate(seq_sizes):
        card = _rand_card_id()
        device = _rand_device_id()
        acct_id = _rand_account_id()
        region = random.choice(INDIAN_CITIES)
        seq_start = base_date + timedelta(
            days=seq_idx,
            hours=random.randint(8, 20),
            minutes=random.randint(0, 59),
        )
        created_at = seq_start - timedelta(minutes=random.randint(2, 30))

        for j in range(small_count):
            txn_id = _rand_txn_id(txn_counter)
            txn_counter += 1
            ts = seq_start + timedelta(minutes=random.uniform(0, 3))
            amount = round(random.uniform(1, 20), 2)
            rows.append({
                "txn_id": txn_id,
                "card_fingerprint": card,
                "device_id": device,
                "ip_region": region,
                "account_id": acct_id,
                "account_created_at": created_at,
                "amount": amount,
                "timestamp": ts,
                "billing_region": region,
                "status": "approved",
                "decline_type": "",
                "true_label": True,
            })

        txn_id = _rand_txn_id(txn_counter)
        txn_counter += 1
        large_ts = seq_start + timedelta(minutes=random.uniform(3.5, 5))
        large_amount = round(random.uniform(2000, 15000), 2)
        rows.append({
            "txn_id": txn_id,
            "card_fingerprint": card,
            "device_id": device,
            "ip_region": region,
            "account_id": acct_id,
            "account_created_at": created_at,
            "amount": large_amount,
            "timestamp": large_ts,
            "billing_region": region,
            "status": "approved",
            "decline_type": "",
            "true_label": True,
        })

    df = pd.DataFrame(rows)
    return df


def generate_ring_abuse_fraud(n=15):
    """2-3 rings of 3-5 accounts sharing device/card/address signals.
    All ring transactions get true_label=True.
    Writes account_relationships.csv with ground truth edges."""
    relationships = []
    rows = []
    txn_counter = 8000

    rings = [
        {"size": 4, "signal_type": "device"},
        {"size": 5, "signal_type": "card"},
        {"size": 3, "signal_type": "address"},
    ]

    base_date = fake.date_time_between(start_date="-15d", end_date="-5d")

    all_ring_accounts = []
    for ring in rings:
        size = ring["size"]
        signal_type = ring["signal_type"]

        if signal_type == "device":
            shared_signal = _rand_device_id()
            ring_accounts = []
            for _ in range(size):
                ring_accounts.append({
                    "account_id": _rand_account_id(),
                    "device_id": shared_signal,
                    "card": _rand_card_id(),
                    "billing_region": random.choice(INDIAN_CITIES),
                })
        elif signal_type == "card":
            shared_signal = _rand_card_id()
            ring_accounts = []
            for _ in range(size):
                ring_accounts.append({
                    "account_id": _rand_account_id(),
                    "device_id": _rand_device_id(),
                    "card": shared_signal,
                    "billing_region": random.choice(INDIAN_CITIES),
                })
        else:
            shared_addr_id = "ADDR_" + fake.hexify(text="^^^^^^^^")
            shared_signal = shared_addr_id
            billing = random.choice(INDIAN_CITIES)
            ring_accounts = []
            for _ in range(size):
                ring_accounts.append({
                    "account_id": _rand_account_id(),
                    "device_id": _rand_device_id(),
                    "card": _rand_card_id(),
                    "billing_region": billing,
                    "shared_address_id": shared_addr_id,
                })

        for i in range(size):
            for j in range(i + 1, size):
                relationships.append({
                    "account_id_a": ring_accounts[i]["account_id"],
                    "account_id_b": ring_accounts[j]["account_id"],
                    "shared_signal": signal_type,
                })

        all_ring_accounts.extend(ring_accounts)

    random.shuffle(all_ring_accounts)
    total_accounts = len(all_ring_accounts)
    base_per_acct = n // total_accounts
    extra = n - base_per_acct * total_accounts

    for i, acct in enumerate(all_ring_accounts):
        created_at = base_date - timedelta(days=random.randint(30, 180))
        txns_for_this = base_per_acct + (1 if i < extra else 0)
        for _ in range(txns_for_this):
            rows.append({
                "txn_id": _rand_txn_id(txn_counter),
                "card_fingerprint": acct["card"],
                "device_id": acct["device_id"],
                "ip_region": random.choice(INDIAN_CITIES),
                "account_id": acct["account_id"],
                "account_created_at": created_at,
                "amount": round(random.uniform(300, 5000), 2),
                "timestamp": base_date + timedelta(
                    days=random.randint(0, 7),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                ),
                "billing_region": acct["billing_region"],
                "status": "approved",
                "decline_type": "",
                "true_label": True,
            })
            txn_counter += 1

    df = pd.DataFrame(rows)
    rel_df = pd.DataFrame(relationships)
    rel_path = os.path.join(DATA_DIR, "account_relationships.csv")
    rel_df.to_csv(rel_path, index=False)
    print(f"Wrote {len(rel_df)} relationship rows to {rel_path}")
    return df


def generate_hard_negatives(n=18):
    """Legitimate but suspicious-looking transactions, true_label=False.
    - Bulk gift-card buyer (5-6 txns, same card, medium amounts R500-1000)
    - Traveling customer (billing_region != ip_region, single normal txn)
    - New account large first purchase (recent account, one txn, no velocity)
    """
    rows = []
    txn_counter = 7000

    # --- Bulk gift-card buyer: 6 txns, same card, rapid succession ---
    gift_card_count = 6
    gift_card_acct = _rand_account_id()
    gift_card_card = _rand_card_id()
    gift_card_device = _rand_device_id()
    gift_card_region = random.choice(INDIAN_CITIES)
    gift_card_created = fake.date_time_between(start_date="-60d", end_date="-7d")
    gift_base = fake.date_time_between(start_date="-10d", end_date="-2d")

    for i in range(gift_card_count):
        rows.append({
            "txn_id": _rand_txn_id(txn_counter),
            "card_fingerprint": gift_card_card,
            "device_id": gift_card_device,
            "ip_region": gift_card_region,
            "account_id": gift_card_acct,
            "account_created_at": gift_card_created,
            "amount": round(random.uniform(500, 1000), 2),
            "timestamp": gift_base + timedelta(minutes=random.uniform(0, 15)),
            "billing_region": gift_card_region,
            "status": "approved",
            "decline_type": "",
            "true_label": False,
        })
        txn_counter += 1

    remaining = n - gift_card_count

    # --- Traveling customers: single transactions with geo mismatch ---
    travel_count = 5
    for i in range(travel_count):
        billing = random.choice(INDIAN_CITIES)
        ip = random.choice([c for c in INDIAN_CITIES if c != billing])
        created = fake.date_time_between(start_date="-90d", end_date="-30d")
        rows.append({
            "txn_id": _rand_txn_id(txn_counter),
            "card_fingerprint": _rand_card_id(),
            "device_id": _rand_device_id(),
            "ip_region": ip,
            "account_id": _rand_account_id(),
            "account_created_at": created,
            "amount": round(random.uniform(100, 6000), 2),
            "timestamp": created + timedelta(
                days=random.randint(7, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            ),
            "billing_region": billing,
            "status": "approved",
            "decline_type": "",
            "true_label": False,
        })
        txn_counter += 1

    # --- New account large first purchase ---
    new_acct_count = remaining - travel_count
    for i in range(new_acct_count):
        created = fake.date_time_between(start_date="-3d", end_date="-1h")
        rows.append({
            "txn_id": _rand_txn_id(txn_counter),
            "card_fingerprint": _rand_card_id(),
            "device_id": _rand_device_id(),
            "ip_region": random.choice(INDIAN_CITIES),
            "account_id": _rand_account_id(),
            "account_created_at": created,
            "amount": round(random.uniform(3000, 12000), 2),
            "timestamp": created + timedelta(minutes=random.randint(5, 120)),
            "billing_region": random.choice(INDIAN_CITIES),
            "status": "approved",
            "decline_type": "",
            "true_label": False,
        })
        txn_counter += 1

    return pd.DataFrame(rows)


def tag_decline_metadata(df):
    """For a subset of non-fraud transactions, set status=declined and
    decline_type to soft_decline (timeout, insufficient funds) or
    hard_decline (expired/stolen card). Fraud transactions stay approved."""
    approved_mask = (df["status"] == "approved") & (df["true_label"] == False)
    candidates = df[approved_mask].index

    num_decline = max(1, int(len(candidates) * 0.08))
    rng = np.random.RandomState(SEED)
    decline_indices = rng.choice(candidates, size=num_decline, replace=False)

    soft_types = ["insufficient_funds", "timeout"]
    hard_types = ["expired_card", "stolen_card"]

    for idx in decline_indices:
        if rng.random() < 0.7:
            df.at[idx, "status"] = "declined"
            df.at[idx, "decline_type"] = rng.choice(soft_types)
        else:
            df.at[idx, "status"] = "declined"
            df.at[idx, "decline_type"] = rng.choice(hard_types)

    return df


def main():
    print("Generating Sentinel synthetic dataset...")
    print(f"Random seed: {SEED}\n")

    # 1. Generate all four categories
    legit_df = generate_legitimate_transactions(250)
    card_test_df = generate_card_testing_fraud(35)
    ring_df = generate_ring_abuse_fraud(15)
    hard_neg_df = generate_hard_negatives(18)

    # 2. Combine into one DataFrame
    combined = pd.concat([legit_df, card_test_df, ring_df, hard_neg_df], ignore_index=True)

    # 3. Ensure correct dtypes
    combined["amount"] = combined["amount"].astype(float)
    combined["true_label"] = combined["true_label"].astype(bool)

    # 4. Build group_ids so related rows stay together in the split
    #    - card-testing sequences: group by card_fingerprint
    #    - ring-abuse accounts: group by ring (connected components from relationships)
    #    - everything else: each row is its own group

    # Identify ring accounts from account_relationships.csv
    rel_path = os.path.join(DATA_DIR, "account_relationships.csv")
    ring_account_to_group = {}
    if os.path.exists(rel_path):
        rel_df = pd.read_csv(rel_path)
        # Build adjacency and find connected components via union-find
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for _, row in rel_df.iterrows():
            a, b = row["account_id_a"], row["account_id_b"]
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            union(a, b)

        # Map each account to its component root
        all_ring_accounts = set(rel_df["account_id_a"]) | set(rel_df["account_id_b"])
        roots = {}
        for acct in all_ring_accounts:
            root = find(acct)
            roots.setdefault(root, [])
            roots[root].append(acct)

        # Assign ring group ids
        for ring_id, accounts in roots.items():
            for acct in accounts:
                ring_account_to_group[acct] = f"ring_{ring_id}"

    # Identify card-testing sequences: true_label=True cards with >1 txn
    fraud_mask = combined["true_label"] == True
    card_test_counts = combined.loc[fraud_mask].groupby("card_fingerprint").size()
    card_test_cards = set(card_test_counts[card_test_counts > 1].index)

    # Assign group_id to each row
    def assign_group(row):
        if row["account_id"] in ring_account_to_group:
            return ring_account_to_group[row["account_id"]]
        if row["true_label"] and row["card_fingerprint"] in card_test_cards:
            return f"card_{row['card_fingerprint']}"
        return f"single_{row['txn_id']}"

    combined["group_id"] = combined.apply(assign_group, axis=1)

    # Debug: verify ring group assignment for ACCT_2643435e
    target_ring_acct = "ACCT_2643435e"
    if target_ring_acct in ring_account_to_group:
        target_ring_group = ring_account_to_group[target_ring_acct]
        ring_members = [a for a, g in ring_account_to_group.items() if g == target_ring_group]
        print(f"\n=== Ring Group Debug: {target_ring_group} ===")
        print(f"  Accounts in ring: {sorted(ring_members)}")
        for acct in sorted(ring_members):
            acct_rows = combined[combined["account_id"] == acct]
            grp = acct_rows["group_id"].iloc[0] if len(acct_rows) > 0 else "NOT IN DATA"
            print(f"  {acct}: group_id={grp}, txns={len(acct_rows)}")

    # 5. Split by group: get unique group_ids, split those, then expand back
    group_labels = combined.groupby("group_id")["true_label"].first()

    unique_groups = group_labels.index.tolist()
    unique_labels = group_labels.values

    grps_train, grps_test = train_test_split(
        unique_groups, test_size=0.3, random_state=SEED, stratify=unique_labels
    )

    train_groups = set(grps_train)
    test_groups = set(grps_test)

    train_df = combined[combined["group_id"].isin(train_groups)].drop(columns=["group_id"])
    test_df = combined[combined["group_id"].isin(test_groups)].drop(columns=["group_id"])

    # 5b. Tag decline metadata on each split independently
    tag_decline_metadata(train_df)
    tag_decline_metadata(test_df)

    # 6. Validation: confirm no card_fingerprint or ring is split across both
    print("=== Split Validation ===")
    violations = 0

    # Check card-testing sequences
    for card in card_test_cards:
        in_train = card in train_df["card_fingerprint"].values
        in_test = card in test_df["card_fingerprint"].values
        if in_train and in_test:
            print(f"  VIOLATION: card {card} split across train and test!")
            violations += 1

    # Check ring accounts
    for ring_id, accounts in roots.items():
        train_accts = set(train_df["account_id"])
        test_accts = set(test_df["account_id"])
        in_train = any(a in train_accts for a in accounts)
        in_test = any(a in test_accts for a in accounts)
        if in_train and in_test:
            print(f"  VIOLATION: ring {ring_id} split across train and test!")
            violations += 1

    if violations == 0:
        print("  All card-testing sequences and rings are intact within their split.")

    # 7. Save CSVs
    train_path = os.path.join(DATA_DIR, "transactions_train.csv")
    test_path = os.path.join(DATA_DIR, "transactions_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total rows:       {len(combined)}")
    print(f"  Legitimate:     {len(legit_df)}")
    print(f"  Card-testing:   {len(card_test_df)}")
    print(f"  Ring-abuse:     {len(ring_df)}")
    print(f"  Hard negatives: {len(hard_neg_df)}")
    print(f"\nTrain split:      {len(train_df)} rows")
    print(f"  Fraud ratio:    {train_df['true_label'].mean():.2%}")
    print(f"Test split:       {len(test_df)} rows")
    print(f"  Fraud ratio:    {test_df['true_label'].mean():.2%}")
    print(f"\nSaved: {train_path}")
    print(f"Saved: {test_path}")

    print(f"\n=== Decline Metadata Verification ===")
    train_declined = (train_df["status"] == "declined").sum()
    test_declined = (test_df["status"] == "declined").sum()
    print(f"  Train declined: {train_declined} / {len(train_df)}")
    print(f"  Test  declined: {test_declined} / {len(test_df)}")


if __name__ == "__main__":
    main()
