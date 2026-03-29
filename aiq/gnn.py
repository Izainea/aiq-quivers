"""
Algebraic Attention and Over-Squashing Analysis for Graph Neural Networks.

This module connects the Impact Automaton on Quivers (IAQ) framework with
Graph Neural Network architectures, providing:

  - AttentionQuiver: constructs a quiver from multi-head attention patterns
  - Path entropy: information-theoretic analysis of walk distributions
  - Algebraic pruning: quotient algebra kQ/I for redundancy reduction
  - PyG bridge: conversion between aiq Quiver ↔ PyG Data objects

Reference: ACT_en.tex, Section 4.4 (Connection with GNNs) and
           Remark on over-squashing and quotient algebras.
"""

from __future__ import annotations

from typing import Optional, Callable
import warnings

import numpy as np

from .quiver import Quiver
from .path_algebra import PathAlgebra, PathAlgebraElement, Ideal, QuotientAlgebra
from .impact import impact_vector, impact_vector_matrix


# ═══════════════════════════════════════════════════════════════════════════
# Path entropy
# ═══════════════════════════════════════════════════════════════════════════

def path_entropy(quiver: Quiver, source, target, k: int,
                 weights: Optional[np.ndarray] = None) -> float:
    """
    Shannon entropy of the walk distribution of length k from source to target.

    When all walks have equal weight, H_k(i,j) = log2(n_k) where n_k is the
    number of walks (i.e., (A^k)_{ij}).  With non-uniform weights, each walk
    p contributes w(p)/W to the distribution.

    Parameters
    ----------
    quiver : Quiver
    source, target : vertex labels
    k : int — walk length
    weights : np.ndarray, optional
        If given, shape (n, n) weight matrix applied per step.
        If None, uniform weights (all walks contribute equally).

    Returns
    -------
    float — H_k(i,j) in bits (log base 2).  Returns 0 if no walks exist.
    """
    algebra = PathAlgebra(quiver)
    walks = algebra.paths_from_to(source, target, k)
    n_walks = len(walks)

    if n_walks == 0:
        return 0.0

    if weights is None:
        # Uniform: H = log2(n_walks)
        return np.log2(n_walks) if n_walks > 1 else 0.0

    # Weighted: compute weight of each walk via product of edge weights
    walk_weights = []
    for path in walks:
        w = 1.0
        for arrow_name in path.arrows:
            si = quiver.vertex_index(quiver.source(arrow_name))
            ti = quiver.vertex_index(quiver.target(arrow_name))
            w *= weights[si, ti]
        walk_weights.append(w)

    total = sum(walk_weights)
    if total <= 0:
        return 0.0

    probs = [w / total for w in walk_weights if w > 0]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def path_entropy_matrix(quiver: Quiver, max_k: int,
                        weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute path entropy H_k(i,j) for all pairs (i,j) and k = 0, ..., max_k.

    Returns
    -------
    np.ndarray of shape (n, n, max_k + 1)
    """
    n = quiver.n_vertices
    result = np.zeros((n, n, max_k + 1))
    for k in range(max_k + 1):
        for i, vi in enumerate(quiver.Q0):
            for j, vj in enumerate(quiver.Q0):
                result[i, j, k] = path_entropy(quiver, vi, vj, k, weights)
    return result


def over_squashing_diagnostic(quiver: Quiver, max_k: int,
                              representation_dim: int) -> dict:
    """
    Identify (source, target, depth) triples where over-squashing occurs.

    Over-squashing happens when H_k(i,j) > log2(d), meaning the walk
    distribution carries more information than the representation can encode.

    Parameters
    ----------
    quiver : Quiver
    max_k : int — maximum depth to check
    representation_dim : int — dimension d of node representations

    Returns
    -------
    dict with keys:
        'capacity': float — log2(d)
        'bottlenecks': list of (source, target, k, H_k) where H_k > log2(d)
        'max_entropy': (source, target, k, H_k) — worst bottleneck
        'walk_counts': np.ndarray — (A^k)_{ij} for reference
    """
    capacity = np.log2(representation_dim)
    bottlenecks = []
    max_entry = (None, None, 0, 0.0)
    n = quiver.n_vertices

    iv_matrix = impact_vector_matrix(quiver, max_k=max_k)  # (n, n, max_k+1)

    for k in range(1, max_k + 1):
        for i, vi in enumerate(quiver.Q0):
            for j, vj in enumerate(quiver.Q0):
                n_walks = int(iv_matrix[i, j, k])
                if n_walks <= 1:
                    continue
                h_k = np.log2(n_walks)  # uniform entropy
                if h_k > capacity:
                    bottlenecks.append((vi, vj, k, h_k))
                if h_k > max_entry[3]:
                    max_entry = (vi, vj, k, h_k)

    return {
        "capacity": capacity,
        "bottlenecks": bottlenecks,
        "max_entropy": max_entry,
        "walk_counts": iv_matrix,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Attention Quiver — multi-head attention as parallel arrows
# ═══════════════════════════════════════════════════════════════════════════

class AttentionQuiver:
    """
    Constructs a quiver from multi-head attention patterns.

    In a Graph Attention Network (GAT) with H heads, each head attending
    from node u to node v produces a distinct arrow α_{uv}^{(h)} in the
    quiver.  Multiple heads = parallel arrows — exactly the quiver structure.

    After L layers, the composed attention from u to v lives in the graded
    component e_u · kQ_L · e_v of the path algebra.  Its dimension counts
    the number of distinct "attention paths" of depth L.

    Reference: ACT_en.tex, Definition 5.2 (Attention Quiver) and
               Remark on head pruning as ideal quotient.
    """

    def __init__(self, quiver: Quiver):
        self._quiver = quiver
        self._algebra = PathAlgebra(quiver)

    @property
    def quiver(self) -> Quiver:
        return self._quiver

    @property
    def algebra(self) -> PathAlgebra:
        return self._algebra

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def from_attention_weights(
        cls,
        nodes: list,
        attention: np.ndarray,
        threshold: float = 0.0,
    ) -> "AttentionQuiver":
        """
        Build an AttentionQuiver from a multi-head attention matrix.

        Parameters
        ----------
        nodes : list
            Node labels.
        attention : np.ndarray
            Shape (n, n, H) where attention[i, j, h] is the attention weight
            from node i to node j in head h.  Or shape (n, n) for single-head.
        threshold : float
            Minimum attention weight to create an arrow.

        Returns
        -------
        AttentionQuiver
        """
        if attention.ndim == 2:
            attention = attention[:, :, np.newaxis]
        n, _, H = attention.shape

        arrows = []
        weights = {}
        for h in range(H):
            for i in range(n):
                for j in range(n):
                    if attention[i, j, h] > threshold:
                        name = f"a_{nodes[i]}_{nodes[j]}_h{h}"
                        arrows.append((name, nodes[i], nodes[j]))
                        weights[name] = attention[i, j, h]

        q = Quiver(nodes, arrows)
        aq = cls(q)
        aq._attention_weights = weights
        aq._n_heads = H
        return aq

    @classmethod
    def from_edge_index(
        cls,
        edge_index: np.ndarray,
        n_nodes: int,
        n_heads: int = 1,
        node_labels: Optional[list] = None,
    ) -> "AttentionQuiver":
        """
        Build an AttentionQuiver from a PyG-style edge_index.

        Parameters
        ----------
        edge_index : np.ndarray
            Shape (2, E) — source and target indices.
        n_nodes : int
        n_heads : int
            Number of attention heads; creates parallel arrows per edge.
        node_labels : list, optional
            Labels for nodes; defaults to [0, 1, ..., n_nodes-1].

        Returns
        -------
        AttentionQuiver
        """
        labels = node_labels or list(range(n_nodes))
        arrows = []
        for h in range(n_heads):
            for e in range(edge_index.shape[1]):
                src, tgt = int(edge_index[0, e]), int(edge_index[1, e])
                name = f"a_{labels[src]}_{labels[tgt]}_h{h}"
                arrows.append((name, labels[src], labels[tgt]))

        q = Quiver(labels, arrows)
        aq = cls(q)
        aq._n_heads = n_heads
        return aq

    # ── Analysis ────────────────────────────────────────────────────────

    def attention_paths(self, source, target, depth: int) -> list:
        """
        Enumerate all attention paths of given depth from source to target.

        Each path is a sequence of arrows, potentially using different heads
        at each step.  The count equals dim(e_source · kQ_depth · e_target).
        """
        return self._algebra.paths_from_to(source, target, depth)

    def attention_path_count(self, source, target, depth: int) -> int:
        """Number of distinct attention paths of given depth."""
        return self._algebra.dimension(source, target, depth)

    def path_entropy(self, source, target, depth: int) -> float:
        """Shannon entropy of the uniform walk distribution at given depth."""
        return path_entropy(self._quiver, source, target, depth)

    def over_squashing_report(self, max_depth: int,
                              representation_dim: int) -> dict:
        """Diagnose over-squashing across all pairs and depths."""
        return over_squashing_diagnostic(
            self._quiver, max_depth, representation_dim
        )

    # ── Algebraic pruning ───────────────────────────────────────────────

    def find_redundant_heads(self, depth: int = 1) -> list:
        """
        Identify pairs of parallel arrows (attention heads) that connect
        the same source-target pair at the given depth.

        Returns list of (arrow_name_1, arrow_name_2, source, target) tuples
        representing potentially redundant head pairs.
        """
        from collections import defaultdict
        # Group arrows by (source, target)
        edge_groups = defaultdict(list)
        for name, src, tgt in self._quiver.Q1:
            edge_groups[(src, tgt)].append(name)

        redundant = []
        for (src, tgt), arrow_names in edge_groups.items():
            if len(arrow_names) >= 2:
                for i in range(len(arrow_names)):
                    for j in range(i + 1, len(arrow_names)):
                        redundant.append(
                            (arrow_names[i], arrow_names[j], src, tgt)
                        )
        return redundant

    def prune_by_ideal(
        self,
        relations: list[tuple[str, str]],
    ) -> QuotientAlgebra:
        """
        Construct the quotient algebra kQ/I that identifies redundant paths.

        Parameters
        ----------
        relations : list of (arrow_name_1, arrow_name_2)
            Pairs of arrows to identify (i.e., generate I by their difference).

        Returns
        -------
        QuotientAlgebra — the pruned algebra kQ/I.
        """
        generators = []
        for a1_name, a2_name in relations:
            elem1 = self._algebra.arrow_element(a1_name)
            elem2 = self._algebra.arrow_element(a2_name)
            generators.append(elem1 - elem2)

        ideal = Ideal(self._algebra, generators)
        return QuotientAlgebra(self._algebra, ideal)

    def pruning_analysis(self, relations: list[tuple[str, str]],
                         max_depth: int) -> dict:
        """
        Compare dimensions and path entropy before and after pruning.

        Returns
        -------
        dict with keys:
            'original_dims': dict (source, target, k) → dim(e_i · kQ_k · e_j)
            'pruned_dims': dict (source, target, k) → dim(e_i · (kQ/I)_k · e_j)
            'reduction': dict (source, target, k) → original - pruned
            'quotient': QuotientAlgebra
        """
        quotient = self.prune_by_ideal(relations)
        original = {}
        pruned = {}
        reduction = {}

        for k in range(1, max_depth + 1):
            for vi in self._quiver.Q0:
                for vj in self._quiver.Q0:
                    key = (vi, vj, k)
                    d_orig = self._algebra.dimension(vi, vj, k)
                    d_pruned = quotient.dimension(vi, vj, k)
                    original[key] = d_orig
                    pruned[key] = d_pruned
                    reduction[key] = d_orig - d_pruned

        return {
            "original_dims": original,
            "pruned_dims": pruned,
            "reduction": reduction,
            "quotient": quotient,
        }


# ═══════════════════════════════════════════════════════════════════════════
# GNN message-passing comparison
# ═══════════════════════════════════════════════════════════════════════════

def algebraic_aggregation(quiver: Quiver, features: dict,
                          max_depth: int,
                          P: Optional[Callable[[int], float]] = None,
                          quotient: Optional[QuotientAlgebra] = None
                          ) -> dict:
    """
    Compute the algebraic aggregation (enriched impact rate style) for each
    node, analogous to multi-layer GNN message passing.

    For each node v and depth g, aggregates features from nodes at distance g
    weighted by the number of walks (A^g)_{uv} and impact weight P(g).

    Parameters
    ----------
    quiver : Quiver
    features : dict
        {vertex: np.ndarray} — feature vector for each node.
    max_depth : int
        Maximum depth (analogous to number of GNN layers).
    P : callable, optional
        Impact weight function P(g).  Defaults to P(g) = 1/(g+1).
    quotient : QuotientAlgebra, optional
        If provided, uses effective walk counts from kQ/I.

    Returns
    -------
    dict — {vertex: np.ndarray} aggregated features.
    """
    if P is None:
        P = lambda g: 1.0 / (g + 1)

    n = quiver.n_vertices
    vertices = quiver.Q0

    # Stack features into matrix (n x d)
    d = len(next(iter(features.values())))
    F = np.zeros((n, d))
    for v in vertices:
        F[quiver.vertex_index(v)] = features[v]

    result = {}
    for v in vertices:
        vi = quiver.vertex_index(v)
        aggregated = np.zeros(d)

        for g in range(1, max_depth + 1):
            if quotient is not None:
                M = quotient.effective_walk_matrix(g)
            else:
                M = quiver.adjacency_power(g)

            # Weighted sum of features from all nodes, weighted by walks to v
            walk_weights = M[:, vi]  # column vi: walks from each node to v
            total_weight = walk_weights.sum()

            if total_weight > 0:
                weighted_features = (walk_weights[:, np.newaxis] * F).sum(axis=0)
                aggregated += P(g) * weighted_features / total_weight

        result[v] = aggregated

    return result


def compare_with_gnn(quiver: Quiver, features: dict,
                     max_depth: int,
                     P: Optional[Callable[[int], float]] = None,
                     quotient: Optional[QuotientAlgebra] = None) -> dict:
    """
    Compare algebraic aggregation (via path algebra) with standard GNN
    message-passing aggregation (mean aggregation, like GCN).

    The GNN-style aggregation uses iterative 1-hop averaging:
        h_v^{(k+1)} = mean({h_u^{(k)} : u → v})

    The algebraic aggregation uses the full path algebra structure:
        agg_v = Σ_g P(g) · (Σ_u (A^g)_{uv} h_u) / (Σ_u (A^g)_{uv})

    Returns
    -------
    dict with keys:
        'algebraic': {vertex: features} — path algebra aggregation
        'gnn_iterative': {vertex: features_per_layer} — GNN layer outputs
        'difference': {vertex: np.ndarray} — difference at final layer
    """
    n = quiver.n_vertices
    vertices = quiver.Q0
    d = len(next(iter(features.values())))

    # Algebraic aggregation
    alg_result = algebraic_aggregation(quiver, features, max_depth, P, quotient)

    # GNN-style iterative mean aggregation
    A = quiver.adjacency_matrix().astype(float)
    # Normalize: D^{-1} A (row normalization for in-neighbors)
    in_deg = A.sum(axis=0)  # column sums = in-degree
    in_deg[in_deg == 0] = 1.0
    A_norm = A / in_deg[np.newaxis, :]  # normalize columns

    F = np.zeros((n, d))
    for v in vertices:
        F[quiver.vertex_index(v)] = features[v]

    gnn_layers = [F.copy()]
    H = F.copy()
    for layer in range(max_depth):
        H = A_norm.T @ H  # message passing: each node averages in-neighbors
        gnn_layers.append(H.copy())

    # Final GNN output per node
    gnn_result = {}
    for v in vertices:
        vi = quiver.vertex_index(v)
        gnn_result[v] = {k: gnn_layers[k][vi] for k in range(max_depth + 1)}

    # Difference at final depth
    diff = {}
    for v in vertices:
        vi = quiver.vertex_index(v)
        diff[v] = alg_result[v] - gnn_layers[max_depth][vi]

    return {
        "algebraic": alg_result,
        "gnn_iterative": gnn_result,
        "difference": diff,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PyG bridge — conversion between aiq Quiver and PyG Data
# ═══════════════════════════════════════════════════════════════════════════

def quiver_to_pyg(quiver: Quiver, node_features: Optional[dict] = None):
    """
    Convert an aiq Quiver to a PyTorch Geometric Data object.

    Parameters
    ----------
    quiver : Quiver
    node_features : dict, optional
        {vertex: np.ndarray} — features for each node.
        If None, uses one-hot encoding.

    Returns
    -------
    torch_geometric.data.Data

    Raises
    ------
    ImportError if torch or torch_geometric is not installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for this function. "
            "Install with: pip install torch torch_geometric"
        )

    n = quiver.n_vertices
    vertices = quiver.Q0

    # Edge index
    src_indices = []
    tgt_indices = []
    for name, src, tgt in quiver.Q1:
        src_indices.append(quiver.vertex_index(src))
        tgt_indices.append(quiver.vertex_index(tgt))

    edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)

    # Node features
    if node_features is not None:
        d = len(next(iter(node_features.values())))
        x = torch.zeros(n, d, dtype=torch.float)
        for v in vertices:
            x[quiver.vertex_index(v)] = torch.tensor(
                node_features[v], dtype=torch.float
            )
    else:
        x = torch.eye(n, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = n
    data.vertex_labels = vertices
    return data


def pyg_to_quiver(data, node_labels: Optional[list] = None) -> Quiver:
    """
    Convert a PyTorch Geometric Data object to an aiq Quiver.

    Parameters
    ----------
    data : torch_geometric.data.Data
    node_labels : list, optional
        Labels for nodes.  Defaults to integer indices.

    Returns
    -------
    Quiver
    """
    edge_index = data.edge_index.cpu().numpy()
    n = data.num_nodes
    labels = node_labels or list(range(n))

    # Detect parallel edges by counting (src, tgt) duplicates
    from collections import Counter
    edge_counts = Counter()
    arrows = []
    for e in range(edge_index.shape[1]):
        src, tgt = int(edge_index[0, e]), int(edge_index[1, e])
        edge_counts[(src, tgt)] += 1
        idx = edge_counts[(src, tgt)]
        name = f"e_{labels[src]}_{labels[tgt]}_{idx}"
        arrows.append((name, labels[src], labels[tgt]))

    return Quiver(labels, arrows)


def run_pyg_gat(data, hidden_dim: int = 16, n_heads: int = 4,
                n_layers: int = 2, return_attention: bool = True):
    """
    Run a Graph Attention Network on a PyG Data object and return
    the output representations and (optionally) attention weights.

    Parameters
    ----------
    data : torch_geometric.data.Data
    hidden_dim : int
    n_heads : int
    n_layers : int
    return_attention : bool

    Returns
    -------
    dict with keys:
        'output': np.ndarray (n, out_dim)
        'attention': list of np.ndarray (E, H) per layer (if return_attention)
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv
    except ImportError:
        raise ImportError(
            "PyTorch and PyTorch Geometric are required. "
            "Install with: pip install torch torch_geometric"
        )

    in_dim = data.x.shape[1]

    # Build a simple GAT
    layers = []
    dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [hidden_dim]
    for i in range(n_layers):
        layers.append(
            GATConv(dims[i] if i == 0 else dims[i] * n_heads,
                    dims[i + 1],
                    heads=n_heads,
                    concat=(i < n_layers - 1))
        )

    # Forward pass collecting attention
    x = data.x
    edge_index = data.edge_index
    attention_weights = []

    with torch.no_grad():
        for i, layer in enumerate(layers):
            if return_attention:
                x, (ei, alpha) = layer(x, edge_index,
                                       return_attention_weights=True)
                attention_weights.append(alpha.cpu().numpy())
            else:
                x = layer(x, edge_index)
            if i < n_layers - 1:
                x = F.elu(x)

    result = {"output": x.cpu().numpy()}
    if return_attention:
        result["attention"] = attention_weights
    return result


def full_comparison(quiver: Quiver, features: dict,
                    max_depth: int = 2,
                    n_heads: int = 4,
                    hidden_dim: int = 16,
                    pruning_relations: Optional[list] = None):
    """
    Full comparison pipeline: algebraic analysis vs PyG GAT.

    1. Converts quiver to PyG Data
    2. Runs algebraic aggregation (with and without kQ/I)
    3. Runs PyG GAT
    4. Compares outputs and reports over-squashing diagnostics

    Parameters
    ----------
    quiver : Quiver
    features : dict — {vertex: np.ndarray}
    max_depth : int
    n_heads : int — heads for PyG GAT
    hidden_dim : int
    pruning_relations : list of (arrow1, arrow2) pairs, optional

    Returns
    -------
    dict with algebraic results, GNN results, and comparison metrics.
    """
    # Algebraic analysis
    alg_result = algebraic_aggregation(quiver, features, max_depth)

    # With pruning
    quotient = None
    pruned_result = None
    if pruning_relations:
        aq = AttentionQuiver(quiver)
        analysis = aq.pruning_analysis(pruning_relations, max_depth)
        quotient = analysis["quotient"]
        pruned_result = algebraic_aggregation(
            quiver, features, max_depth, quotient=quotient
        )

    # Over-squashing diagnostic
    diag = over_squashing_diagnostic(quiver, max_depth, hidden_dim)

    result = {
        "algebraic": alg_result,
        "algebraic_pruned": pruned_result,
        "over_squashing": diag,
    }

    # PyG comparison (if available)
    try:
        data = quiver_to_pyg(quiver, features)
        gat_result = run_pyg_gat(
            data, hidden_dim=hidden_dim, n_heads=n_heads, n_layers=max_depth
        )
        result["pyg_gat"] = gat_result
    except ImportError:
        result["pyg_gat"] = None
        warnings.warn(
            "PyG not available; skipping GNN comparison. "
            "Install with: pip install torch torch_geometric"
        )

    return result
