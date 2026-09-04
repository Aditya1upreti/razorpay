"""
DAY 3 — Layer 2: Relationship Graph + Ring Detection

Nodes = accounts
Edges = shared device_id, shared billing_region+address, or shared
        card_fingerprint between two accounts

Uses networkx. Community/cluster detection can be as simple as
connected_components() — doesn't need to be fancy for the hackathon.

Also produces a BFS-style plain-English traversal explanation, e.g.:
  "Account A -> shares device with -> Account B -> shares card with -> Account C"
"""

import os
import sys

import networkx as nx
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def build_graph(transactions_df):
    """Build an undirected graph where accounts are nodes and edges
    represent a shared device/address/card between two accounts."""
    G = nx.Graph()

    for acct in transactions_df["account_id"].unique():
        G.add_node(acct)

    rel_path = os.path.join(DATA_DIR, "account_relationships.csv")
    if not os.path.exists(rel_path):
        print(
            "WARNING: Ring detection unavailable: account_relationships.csv not found. "
            "Graph will have no edges — ring detection contributes nothing.",
            file=sys.stderr,
        )
        return G

    rels = pd.read_csv(rel_path)

    for _, row in rels.iterrows():
        a, b, sig = row["account_id_a"], row["account_id_b"], row["shared_signal"]
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b, signal=sig)

    return G


def find_ring_candidates(graph, min_cluster_size=3):
    """Return list of connected components (as sets of account_ids)
    with size >= min_cluster_size — these are your ring candidates."""
    candidates = []
    for component in nx.connected_components(graph):
        if len(component) >= min_cluster_size:
            candidates.append(component)
    return candidates


def explain_ring_bfs(graph, ring_accounts):
    """Walk the ring via BFS from the highest-degree (most-connected)
    node and produce a human-readable step-by-step trail string."""
    subgraph = graph.subgraph(ring_accounts)

    start_node = max(subgraph.nodes, key=lambda n: subgraph.degree(n))

    lines = []
    visited_edges = set()
    for u, v in nx.bfs_edges(subgraph, source=start_node):
        sig = subgraph.edges[u, v]["signal"]
        edge_key = frozenset([u, v])
        if edge_key not in visited_edges:
            visited_edges.add(edge_key)
            lines.append(f"Account {u} -> shares {sig} with -> Account {v}")

    return "\n".join(lines)


if __name__ == "__main__":
    txns = pd.read_csv(os.path.join(DATA_DIR, "transactions_train.csv"))
    graph = build_graph(txns)
    rings = find_ring_candidates(graph)

    print(f"=== Ring Candidates Found: {len(rings)} ===\n")
    all_ring_accounts = set()
    for i, ring in enumerate(rings, 1):
        all_ring_accounts |= ring
        trail = explain_ring_bfs(graph, ring)
        print(f"Ring {i} ({len(ring)} accounts): {sorted(ring)}")
        print(trail)
        print()

    fraud_txns = txns[txns["true_label"] == True]
    missed = fraud_txns[~fraud_txns["account_id"].isin(all_ring_accounts)]

    print(f"=== Fraud Transactions Missed by Ring Detector: {len(missed)} ===")
    if len(missed) > 0:
        for _, row in missed.iterrows():
            print(f"  {row['txn_id']}  account={row['account_id']}  amount={row['amount']}")
    else:
        print("  (none)")
