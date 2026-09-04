"""
Larger synthetic held-out set for confidence calibration analysis.

Generates ~500 transactions with ground-truth fraud/not-fraud labels,
keeping the same realistic distribution as the main dataset. Saved as:
  - data/transactions_calibration.csv
  - data/account_relationships_calibration.csv

This set is SEPARATE from the frozen train/test split and is used
exclusively for calibration analysis in calibration_report.py.

Target composition (~500 total transactions):
  - ~390 legitimate transactions
  - ~55  card-testing fraud sequences
  - ~25  ring-abuse fraud
  - ~30  hard negatives
"""

import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker

SEED_CAL = 137  # Different from main dataset (42)
random.seed(SEED_CAL)
np.random.seed(SEED_CAL)
fake = Faker()
Faker.seed(SEED_CAL)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad",
    "Chennai", "Kolkata", "Pune", "Jaipur", "Lucknow",
    "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal",
    "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra",
]


def _rand_account_id():
    return "ACCTCAL_" + fake.hexify(text="^^^^^^^^")


def _rand_card_id():
    return "CARDCAL_" + fake.hexify(text="^^^^^^^^")


def _rand_device_id():
    return "DEVCAL_" + fake.hexify(text="^^^^^^^^")


def _rand_txn_id(counter):
    return f"TXNCAL{counter:05d}"


def generate_legitimate(n=390):
    rows = []
    txn_counter = 1

    num_accounts = random.randint(80, 120)
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

    return pd.DataFrame(rows)


def generate_card_testing(n=55):
    rows = []
    txn_counter = 9000

    num_sequences = random.randint(7, 9)
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

    return pd.DataFrame(rows)


def generate_ring_abuse(n=25):
    relationships = []
    rows = []
    txn_counter = 8000

    rings = [
        {"size": 4, "signal_type": "device"},
        {"size": 5, "signal_type": "card"},
        {"size": 3, "signal_type": "address"},
        {"size": 4, "signal_type": "device"},
        {"size": 3, "signal_type": "card"},
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
            shared_addr_id = "ADDRCAL_" + fake.hexify(text="^^^^^^^^")
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
    return df, rel_df


def generate_hard_negatives(n=30):
    rows = []
    txn_counter = 7000

    gift_card_count = 8
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

    travel_count = 8
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
    approved_mask = (df["status"] == "approved") & (df["true_label"] == False)
    candidates = df[approved_mask].index

    num_decline = max(1, int(len(candidates) * 0.08))
    rng = np.random.RandomState(SEED_CAL)
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
    print("Generating calibration dataset...")
    print(f"Random seed: {SEED_CAL}\n")

    legit_df = generate_legitimate(390)
    card_test_df = generate_card_testing(55)
    ring_df, rel_df = generate_ring_abuse(25)
    hard_neg_df = generate_hard_negatives(30)

    combined = pd.concat([legit_df, card_test_df, ring_df, hard_neg_df], ignore_index=True)
    combined["amount"] = combined["amount"].astype(float)
    combined["true_label"] = combined["true_label"].astype(bool)

    tag_decline_metadata(combined)

    cal_path = os.path.join(DATA_DIR, "transactions_calibration.csv")
    rel_path = os.path.join(DATA_DIR, "account_relationships_calibration.csv")

    combined.to_csv(cal_path, index=False)
    rel_df.to_csv(rel_path, index=False)

    print("=" * 60)
    print("CALIBRATION DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total rows:       {len(combined)}")
    print(f"  Legitimate:     {len(legit_df)}")
    print(f"  Card-testing:   {len(card_test_df)}")
    print(f"  Ring-abuse:     {len(ring_df)}")
    print(f"  Hard negatives: {len(hard_neg_df)}")
    print(f"  Fraud ratio:    {combined['true_label'].mean():.2%}")
    print(f"\nRelationships:    {len(rel_df)} rows")
    print(f"\nSaved: {cal_path}")
    print(f"Saved: {rel_path}")


if __name__ == "__main__":
    main()
