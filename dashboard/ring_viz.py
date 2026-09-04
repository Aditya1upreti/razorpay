"""
Ring Visualization — Interactive node-edge graph for ring detection.

Read-only layer: takes the existing networkx graph from engine/graph_builder.py
and renders it as an interactive plotly figure. No detection logic is modified.
"""

import networkx as nx
import plotly.graph_objects as go
from engine.masking import tokenize_id


def render_ring_graph(graph, ring_accounts, scored_df=None, dark_mode=False):
    """Build an interactive plotly figure for a ring candidate.

    Args:
        graph: networkx Graph from graph_builder.build_graph()
        ring_accounts: set of account_ids in this ring
        scored_df: optional DataFrame with raw_risk_score per account (for hover)
        dark_mode: if True, use dark-theme colors for background/text

    Returns:
        plotly.graph_objects.Figure
    """
    if not ring_accounts:
        return _empty_figure("No ring connections detected.", dark_mode=dark_mode)

    subgraph = graph.subgraph(ring_accounts)

    if len(subgraph.nodes) == 0:
        return _empty_figure("No ring connections detected.", dark_mode=dark_mode)

    # Layout
    pos = nx.spring_layout(subgraph, seed=42, k=2.0)

    # Node risk: use degree within ring as proxy (higher degree = more connections = higher risk)
    degrees = dict(subgraph.degree())
    max_deg = max(degrees.values()) if degrees else 1

    # Build risk lookup from scored_df if available
    risk_lookup = {}
    if scored_df is not None:
        for _, row in scored_df.iterrows():
            if row["account_id"] in ring_accounts:
                risk_lookup[row["account_id"]] = row.get("raw_risk_score", 0)

    # Node positions and attributes
    node_x, node_y = [], []
    node_text, node_hover = [], []
    node_color, node_size = [], []

    for node in subgraph.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Masked account ID (no raw PII)
        masked_id = tokenize_id(str(node), prefix="ACCT")

        # Risk score
        risk = risk_lookup.get(node, 0)
        deg = degrees[node]

        node_text.append(masked_id)
        node_hover.append(
            f"Account: {masked_id}<br>"
            f"Connections in ring: {deg}<br>"
            f"Risk score: {risk}"
        )

        # Color: higher risk = darker red
        if risk > 75:
            node_color.append("rgb(220, 50, 50)")  # red
            node_size.append(28)
        elif risk > 30:
            node_color.append("rgb(255, 165, 0)")  # orange
            node_size.append(22)
        else:
            node_color.append("rgb(100, 149, 237)")  # cornflower blue
            node_size.append(16)

    # Edges
    edge_x, edge_y = [], []
    edge_labels_x, edge_labels_y, edge_labels_text = [], [], []

    for u, v in subgraph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Edge label at midpoint
        signal = subgraph.edges[u, v].get("signal", "shared attribute")
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        edge_labels_x.append(mx)
        edge_labels_y.append(my)
        edge_labels_text.append(signal)

    # Build figure
    fig = go.Figure()

    edge_line_color = "rgb(80, 80, 80)" if dark_mode else "rgb(180, 180, 180)"
    edge_text_color = "rgb(160, 160, 160)" if dark_mode else "rgb(40, 40, 40)"
    node_border = "rgb(30, 30, 30)" if dark_mode else "white"
    text_c = "#F8FAFC" if dark_mode else "#0F172A"
    plot_bg = "#0F1319" if dark_mode else "#F8FAFC"
    paper_bg = "#0A0E14" if dark_mode else "#FFFFFF"

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color=edge_line_color),
        hoverinfo="none",
        showlegend=False,
    ))

    # Edge labels
    fig.add_trace(go.Scatter(
        x=edge_labels_x, y=edge_labels_y,
        mode="text",
        text=edge_labels_text,
        textposition="top center",
        textfont=dict(size=9, color=edge_text_color),
        hoverinfo="none",
        showlegend=False,
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color=node_border)),
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color=text_c),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
    )

    return fig


def _empty_figure(message, dark_mode=False):
    """Return a simple plotly figure with a centered message."""
    text_c = "#F8FAFC" if dark_mode else "#0F172A"
    plot_bg = "#0F1319" if dark_mode else "#F8FAFC"
    paper_bg = "#0A0E14" if dark_mode else "#FFFFFF"
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color=text_c),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
    )
    return fig
