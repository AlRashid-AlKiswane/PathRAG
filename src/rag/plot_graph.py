"""
plot_graph.py
=============

High-performance, 3D interactive visualization utilities for PathRAG semantic graphs.

This module provides `visualize_graph`, a production-ready function that renders a
NetworkX (di)graph as a Plotly 3D scatter/line figure. It includes guardrails,
structured logging, and sensible defaults for large graphs.

Key features
------------
- Degree-based node sampling when graphs are large (keeps the most connected nodes).
- Optional edge filtering by weight to reduce clutter.
- 3D spring layout with stable seeding for reproducible layouts.
- Configurable node attribute mapping for labels and hover text.
- Safe sizing and coloring based on degree and type.
- Comprehensive error handling with structured logs.

Example
-------
>>> fig = visualize_graph(G, max_nodes=150, min_edge_weight=0.2)
>>> fig.show()
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Hashable, Optional, Union, Iterable

import networkx as nx
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Project logging setup (optional, keep if you rely on your infra logger)
# -----------------------------------------------------------------------------
try:
    MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if MAIN_DIR not in sys.path:
        sys.path.append(MAIN_DIR)
except (ImportError, OSError) as e:
    logging.error("Failed to set up main directory path: %s", e, exc_info=True)

try:
    from src.infra import setup_logging  # noqa: E402
    logger = setup_logging(name="PLOT-GRAPH")
except Exception:  # Fallback to root logger if infra logger unavailable
    raise

def _sample_top_degree_nodes(
    g: Union[nx.Graph, nx.DiGraph],
    k: int
) -> Iterable[Hashable]:
    """
    Return up to k node IDs with highest degree (in-degree + out-degree for DiGraph).
    Degree-based sampling tends to preserve hubs/structure for visualization.

    Parameters
    ----------
    g : nx.Graph | nx.DiGraph
        Graph to sample from.
    k : int
        Maximum number of nodes to return.

    Returns
    -------
    Iterable[Hashable]
        Node identifiers of the sampled subgraph.
    """
    # networkx.degree works on both Graph and DiGraph (sum of in/out for DiGraph)
    # Using nlarges manually to avoid importing heapq for brevity.
    deg = dict(g.degree())
    # Sort only once; for very large graphs, a heap would be marginally faster.
    top = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[: max(k, 0)]
    return [n for n, _ in top]


def visualize_graph(
    g: Union[nx.Graph, nx.DiGraph],
    max_nodes: int = 200,
    *,
    node_label_attr: str = "label",
    node_text_attr: str = "text",
    min_edge_weight: Optional[float] = None,
    edge_weight_attr: str = "weight",
    seed: int = 42,
    title: str = "PathRAG Semantic Graph"
) -> go.Figure:
    """
    Generate a 3D interactive Plotly visualization of a (di)graph.

    Nodes represent document chunks or concepts; edges represent semantic relationships.
    Node color is determined by a "type" label (e.g., 'chunk', 'concept', 'query'),
    defaulting to gray if unknown. Node size is scaled by degree.

    Parameters
    ----------
    g : nx.Graph | nx.DiGraph
        Input graph to visualize. Must be non-empty.
    max_nodes : int, optional
        Maximum number of nodes to display. If the graph is larger, the function
        samples the top-degree nodes to preserve structure. Default is 200.
    node_label_attr : str, optional
        Node attribute key containing the node "type"/label (e.g., 'chunk', 'concept').
        Default is "label".
    node_text_attr : str, optional
        Node attribute key containing human-readable text for hover previews.
        Default is "text".
    min_edge_weight : float | None, optional
        If provided, edges with weight below this threshold (as read from
        `edge_weight_attr`) are filtered out for clarity. Default is None (no filter).
    edge_weight_attr : str, optional
        Edge attribute key for weights, used with `min_edge_weight`. Default is "weight".
    seed : int, optional
        Random seed for layout reproducibility. Default is 42.
    title : str, optional
        Plot title. Default is "PathRAG Semantic Graph".

    Returns
    -------
    plotly.graph_objects.Figure
        A 3D Plotly figure with nodes and edges.

    Raises
    ------
    TypeError
        If `g` is not a NetworkX Graph/DiGraph.
    ValueError
        If the graph is empty or `max_nodes` is invalid.

    Notes
    -----
    - Uses `nx.spring_layout(..., dim=3)` for 3D positions.
    - For very large graphs, consider pre-filtering edges or increasing `min_edge_weight`.
    """
    # --------------------------- Validation ---------------------------
    if not isinstance(g, (nx.Graph, nx.DiGraph)):
        logger.error("visualize_graph received unsupported type: %s", type(g))
        raise TypeError("g must be a networkx Graph or DiGraph")

    if g.number_of_nodes() == 0:
        logger.error("Graph is empty - nothing to visualize")
        raise ValueError("Graph is empty - nothing to visualize")

    if not isinstance(max_nodes, int) or max_nodes <= 0:
        logger.error("Invalid max_nodes: %s (must be positive integer)", max_nodes)
        raise ValueError("max_nodes must be a positive integer")

    try:
        # ---------------------- Sampling (if needed) ----------------------
        original_nodes = g.number_of_nodes()
        if original_nodes > max_nodes:
            sampled_nodes = set(_sample_top_degree_nodes(g, max_nodes))
            g = g.subgraph(sampled_nodes).copy()
            logger.info(
                "Sampled graph from %d to %d nodes (top-degree).",
                original_nodes, g.number_of_nodes()
            )

        # ---------------------- Edge filtering (optional) ----------------------
        if min_edge_weight is not None:
            to_remove = []
            for u, v, data in g.edges(data=True):
                w = data.get(edge_weight_attr)
                if w is None or w < float(min_edge_weight):
                    to_remove.append((u, v))
            if to_remove:
                g.remove_edges_from(to_remove)
                logger.info("Filtered %d edges below weight %.4f.", len(to_remove), float(min_edge_weight))

        # Re-check after filtering: we still want something to show.
        if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
            logger.warning("Graph sparse after filtering/sampling; visualization may be minimal.")

        # --------------------------- Layout (3D) ---------------------------
        # spring_layout with dim=3 often gives decent separation in 3D.
        pos = nx.spring_layout(g, dim=3, seed=seed)
        logger.debug("Computed 3D layout for %d nodes.", len(pos))

        # --------------------------- Build edges ---------------------------
        edge_x, edge_y, edge_z = [], [], []
        for src, tgt in g.edges():
            x0, y0, z0 = pos[src]
            x1, y1, z1 = pos[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(width=1, color="lightgray"),
            hoverinfo="none",
            name="edges",
        )

        # --------------------------- Build nodes ---------------------------
        degrees = dict(g.degree())
        max_degree = max(degrees.values()) if degrees else 1

        node_x, node_y, node_z = [], [], []
        node_text, node_color, node_size = [], [], []

        # Color map for common labels; default gray if unknown
        color_map: Dict[str, str] = {
            "chunk": "blue",
            "concept": "green",
            "query": "red",
        }

        for node, attrs in g.nodes(data=True):
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            label = str(attrs.get(node_label_attr, "unknown"))
            preview = attrs.get(node_text_attr)
            if isinstance(preview, str) and preview:
                preview = (preview[:120] + "…") if len(preview) > 120 else preview
            else:
                preview = label

            d = max(1, degrees.get(node, 1))

            # Size: base 6 + scaled by degree (capped for outliers)
            # Compute node size based on degree, with division-by-zero handling
            try:
                max_degree = max(degrees.values()) if degrees else 1
                size = 6.0 + 14.0 * (d / max_degree)
            except ZeroDivisionError:
                size = 6.0  # fallback size if max_degree is 0

            size = min(size, 24.0)  # clamp to avoid giant markers

            node_text.append(f"<b>Node:</b> {node}<br><b>Degree:</b> {d}<br><b>Type:</b> {label}<br>{preview}")
            node_color.append(color_map.get(label, "gray"))
            node_size.append(size)

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color="black"),
                opacity=0.85,
            ),
            hoverinfo="text",
            text=node_text,
            name="nodes",
        )

        # --------------------------- Figure ---------------------------
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                margin=dict(l=0, r=0, t=50, b=0),
                scene=dict(
                    xaxis=dict(showbackground=False, showgrid=False, zeroline=False, title=""),
                    yaxis=dict(showbackground=False, showgrid=False, zeroline=False, title=""),
                    zaxis=dict(showbackground=False, showgrid=False, zeroline=False, title=""),
                ),
                showlegend=False,
                hovermode="closest",
            ),
        )

        logger.info(
            "Graph visualization created: %d nodes, %d edges.",
            g.number_of_nodes(),
            g.number_of_edges(),
        )
        return fig

    except Exception as e:
        logger.error("Failed to build graph visualization: %s", e, exc_info=True)
        # Re-raise to let caller decide how to handle; they might want a fallback image/page.
        raise
