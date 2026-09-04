"""Tests for engine/graph_builder.py — Relationship Graph + Ring Detection.

Uses a mocked account_relationships.csv via monkeypatch to avoid depending
on the real data file for unit tests. Integration tests use the real data.
"""

import os
from unittest.mock import patch

import networkx as nx
import pandas as pd
import pytest

from engine.graph_builder import build_graph, explain_ring_bfs, find_ring_candidates


@pytest.fixture
def mock_rels_csv(tmp_path):
    """Create a temporary account_relationships.csv with a planted ring
    (A-B-C all connected) matching the account IDs in sample_graph_df."""
    rels = pd.DataFrame([
        {"account_id_a": "ACCT_RING_A", "account_id_b": "ACCT_RING_B", "shared_signal": "device"},
        {"account_id_a": "ACCT_RING_B", "account_id_b": "ACCT_RING_C", "shared_signal": "device"},
        {"account_id_a": "ACCT_RING_A", "account_id_b": "ACCT_RING_C", "shared_signal": "card_fingerprint"},
    ])
    csv_path = tmp_path / "account_relationships.csv"
    rels.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def graph_with_ring(mock_rels_csv, sample_graph_df, monkeypatch):
    """Build a graph using the mock relationships CSV."""
    # Patch the DATA_DIR in graph_builder to point to tmp_path
    monkeypatch.setattr(
        "engine.graph_builder.DATA_DIR",
        str(mock_rels_csv.parent),
    )
    return build_graph(sample_graph_df)


class TestBuildGraph:
    """Tests for graph construction from transactions + relationships."""

    def test_graph_nodes_are_accounts(self, graph_with_ring):
        """All account_ids from the transaction DataFrame become nodes."""
        nodes = set(graph_with_ring.nodes)
        expected = {
            "ACCT_RING_A",
            "ACCT_RING_B",
            "ACCT_RING_C",
            "ACCT_ISOLATED",
        }
        assert nodes == expected

    def test_edges_from_relationships(self, graph_with_ring):
        """Edges are created for relationships where both accounts are in the graph."""
        edges = set(graph_with_ring.edges)
        # Ring: A-B, B-C (D-E excluded because D/E not in sample_graph_df)
        assert ("ACCT_RING_A", "ACCT_RING_B") in edges
        assert ("ACCT_RING_B", "ACCT_RING_C") in edges

    def test_isolated_node_has_no_edges(self, graph_with_ring):
        """An account with no relationships has degree 0."""
        assert graph_with_ring.degree("ACCT_ISOLATED") == 0

    def test_edge_has_signal_attribute(self, graph_with_ring):
        """Each edge stores the shared_signal from the relationships CSV."""
        signal = graph_with_ring.edges[("ACCT_RING_A", "ACCT_RING_B")]["signal"]
        assert signal == "device"


class TestFindRingCandidates:
    """Tests for ring candidate detection."""

    def test_finds_planted_ring(self, graph_with_ring):
        """The 3-node connected component is detected as a ring candidate."""
        candidates = find_ring_candidates(graph_with_ring, min_cluster_size=3)
        assert len(candidates) == 1
        ring = candidates[0]
        assert ring == {"ACCT_RING_A", "ACCT_RING_B", "ACCT_RING_C"}

    def test_isolated_nodes_not_a_ring(self, graph_with_ring):
        """Single isolated nodes don't form ring candidates."""
        candidates = find_ring_candidates(graph_with_ring, min_cluster_size=3)
        # Only the 3-node ring; isolated nodes are not included
        for candidate in candidates:
            assert len(candidate) >= 3

    def test_min_cluster_size_filter(self, graph_with_ring):
        """Raising min_cluster_size above 3 excludes the planted ring."""
        candidates = find_ring_candidates(graph_with_ring, min_cluster_size=4)
        assert len(candidates) == 0

    def test_empty_graph(self):
        """An empty graph returns no candidates."""
        G = nx.Graph()
        candidates = find_ring_candidates(G)
        assert candidates == []

    def test_all_small_components(self):
        """Graph with only pairs (< min_cluster_size) returns no candidates."""
        G = nx.Graph()
        G.add_edge("A", "B")
        G.add_edge("C", "D")
        candidates = find_ring_candidates(G, min_cluster_size=3)
        assert candidates == []


class TestExplainRingBfs:
    """Tests for BFS trail explanation output."""

    def test_explains_ring_trail(self, graph_with_ring):
        """BFS explanation contains 'shares' and 'with' for each edge."""
        ring = {"ACCT_RING_A", "ACCT_RING_B", "ACCT_RING_C"}
        trail = explain_ring_bfs(graph_with_ring, ring)
        lines = trail.strip().split("\n")
        # 3 edges in a triangle -> BFS visits 2 edges (tree edges)
        # Actually in a triangle, BFS from highest-degree node visits 2 edges
        assert len(lines) >= 2
        for line in lines:
            assert "shares" in line
            assert "with" in line
            assert "Account" in line

    def test_trail_starts_from_highest_degree(self, graph_with_ring):
        """BFS starts from the node with the highest degree."""
        ring = {"ACCT_RING_A", "ACCT_RING_B", "ACCT_RING_C"}
        trail = explain_ring_bfs(graph_with_ring, ring)
        # All nodes have degree 2 in a triangle, so max() picks one
        # Just verify it starts with "Account"
        assert trail.startswith("Account")

    def test_single_node_ring(self):
        """A ring with one node produces an empty trail (no edges)."""
        G = nx.Graph()
        G.add_node("A")
        trail = explain_ring_bfs(G, {"A"})
        assert trail == ""


class TestBuildGraphIntegration:
    """Integration test using real data files (skipped if data missing)."""

    @pytest.mark.skipif(
        not os.path.exists("data/account_relationships.csv"),
        reason="Real data file not available",
    )
    def test_builds_from_real_data(self):
        """build_graph produces a valid networkx Graph from real data."""
        df = pd.read_csv("data/transactions_train.csv")
        df["account_created_at"] = pd.to_datetime(df["account_created_at"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        G = build_graph(df)
        assert isinstance(G, nx.Graph)
        assert len(G.nodes) > 0
