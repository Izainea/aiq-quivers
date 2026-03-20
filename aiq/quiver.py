"""
Quiver Q = (Q_0, Q_1, s, t) — Grafos dirigidos con flechas múltiples.

Referencia: ACT.tex, Definición 2.4 (Quiver), Definición 2.7 (Matriz de Adyacencia).
"""

from __future__ import annotations

from collections import deque
import warnings
from typing import Hashable, Optional

import numpy as np


class Quiver:
    """
    Un quiver Q = (Q_0, Q_1, s, t).

    Parameters
    ----------
    vertices : list
        Lista de etiquetas de vértices (hashables).
    arrows : list[tuple]
        Cada tupla es (nombre, fuente, objetivo). Permite flechas múltiples.
    weights : dict | None
        Mapa nombre_flecha → {+1, −1} para quivers con signo (Def. 2.11).
        Si None, todas las flechas tienen peso +1.
    """

    __slots__ = (
        "_vertices", "_v2idx", "_arrows", "_weights",
        "_succ_idx", "_pred_idx", "_succ_labels", "_pred_labels",
        "_in_deg", "_out_deg",
        "_arrow_src", "_arrow_tgt",
        "_adj", "_signed_adj", "_adj_powers", "_signed_powers",
        "_diam", "_dist_matrix", "_is_acyclic",
    )

    def __init__(
        self,
        vertices: list[Hashable],
        arrows: list[tuple],
        weights: Optional[dict] = None,
    ):
        self._vertices = list(vertices)
        self._v2idx: dict[Hashable, int] = {v: i for i, v in enumerate(self._vertices)}
        self._arrows = [(str(name), src, tgt) for name, src, tgt in arrows]
        self._weights = weights or {}

        n = len(self._vertices)

        # Pre-compute adjacency lists and degrees in O(V+E)
        self._succ_idx: list[list[int]] = [[] for _ in range(n)]
        self._pred_idx: list[list[int]] = [[] for _ in range(n)]
        self._succ_labels: dict[Hashable, list[Hashable]] = {v: [] for v in self._vertices}
        self._pred_labels: dict[Hashable, list[Hashable]] = {v: [] for v in self._vertices}
        self._in_deg = np.zeros(n, dtype=np.int32)
        self._out_deg = np.zeros(n, dtype=np.int32)

        # Arrow lookup: name → (src, tgt)
        self._arrow_src: dict[str, Hashable] = {}
        self._arrow_tgt: dict[str, Hashable] = {}

        for name, src, tgt in self._arrows:
            if src not in self._v2idx:
                raise ValueError(f"Fuente '{src}' de flecha '{name}' no está en Q_0")
            if tgt not in self._v2idx:
                raise ValueError(f"Objetivo '{tgt}' de flecha '{name}' no está en Q_0")
            si, ti = self._v2idx[src], self._v2idx[tgt]
            self._succ_idx[si].append(ti)
            self._pred_idx[ti].append(si)
            self._succ_labels[src].append(tgt)
            self._pred_labels[tgt].append(src)
            self._out_deg[si] += 1
            self._in_deg[ti] += 1
            self._arrow_src[name] = src
            self._arrow_tgt[name] = tgt

        # Lazy caches
        self._adj: Optional[np.ndarray] = None
        self._signed_adj: Optional[np.ndarray] = None
        self._adj_powers: dict[int, np.ndarray] = {}
        self._signed_powers: dict[int, np.ndarray] = {}
        self._diam: Optional[int] = None
        self._dist_matrix: Optional[np.ndarray] = None
        self._is_acyclic: Optional[bool] = None

    # ── Propiedades básicas ──────────────────────────────────────────

    @property
    def Q0(self) -> list:
        """Vértices del quiver."""
        return list(self._vertices)

    @property
    def Q1(self) -> list[tuple]:
        """Flechas como (nombre, fuente, objetivo)."""
        return list(self._arrows)

    @property
    def n_vertices(self) -> int:
        return len(self._vertices)

    @property
    def n_arrows(self) -> int:
        return len(self._arrows)

    def vertex_index(self, v: Hashable) -> int:
        """Índice entero de un vértice."""
        return self._v2idx[v]

    def source(self, arrow_name: str) -> Hashable:
        """s(α): fuente de la flecha. O(1)."""
        try:
            return self._arrow_src[arrow_name]
        except KeyError:
            raise KeyError(f"Flecha '{arrow_name}' no encontrada")

    def target(self, arrow_name: str) -> Hashable:
        """t(α): objetivo de la flecha. O(1)."""
        try:
            return self._arrow_tgt[arrow_name]
        except KeyError:
            raise KeyError(f"Flecha '{arrow_name}' no encontrada")

    def arrow_weight(self, arrow_name: str) -> int:
        """Peso w(α) ∈ {+1, −1}. Def. 2.11."""
        return self._weights.get(arrow_name, 1)

    # ── Matrices ─────────────────────────────────────────────────────

    def adjacency_matrix(self) -> np.ndarray:
        """
        Matriz de adyacencia A donde A_{ij} = #flechas de i a j.
        (Definición 2.7)
        """
        if self._adj is None:
            n = self.n_vertices
            A = np.zeros((n, n), dtype=np.float64)
            for _, src, tgt in self._arrows:
                A[self._v2idx[src], self._v2idx[tgt]] += 1
            self._adj = A
        return self._adj.copy()

    def signed_adjacency_matrix(self) -> np.ndarray:
        """
        Matriz de adyacencia con signo W_{ij} = Σ w(α) sobre flechas i→j.
        Para quivers sin signo, coincide con adjacency_matrix().
        """
        if self._signed_adj is None:
            n = self.n_vertices
            W = np.zeros((n, n), dtype=np.float64)
            for name, src, tgt in self._arrows:
                W[self._v2idx[src], self._v2idx[tgt]] += self.arrow_weight(name)
            self._signed_adj = W
        return self._signed_adj.copy()

    def adjacency_power(self, k: int) -> np.ndarray:
        """
        A^k — la entrada (A^k)_{ij} cuenta walks de longitud k de i a j.
        (Proposición 2.5)
        """
        if k not in self._adj_powers:
            if self._adj is None:
                self.adjacency_matrix()
            self._adj_powers[k] = np.linalg.matrix_power(self._adj, k)
        return self._adj_powers[k].copy()

    def signed_adjacency_power(self, k: int) -> np.ndarray:
        """
        W^k — walks de longitud k ponderados por producto de signos.
        """
        if k not in self._signed_powers:
            if self._signed_adj is None:
                self.signed_adjacency_matrix()
            self._signed_powers[k] = np.linalg.matrix_power(self._signed_adj, k)
        return self._signed_powers[k].copy()

    # ── Distancias y diámetro ────────────────────────────────────────

    def _compute_distance_matrix(self):
        """BFS desde cada vértice para calcular distancias dirigidas."""
        n = self.n_vertices
        if n > 5000:
            warnings.warn(
                f"Calculando matriz de distancias {n}×{n} "
                f"({n * n * 8 / 1e9:.1f} GB). Considere usar un subconjunto.",
                stacklevel=2,
            )
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)

        succ = self._succ_idx
        for s in range(n):
            row = dist[s]
            queue = deque([s])
            while queue:
                u = queue.popleft()
                d_next = row[u] + 1
                for v in succ[u]:
                    if row[v] > d_next:
                        row[v] = d_next
                        queue.append(v)
        self._dist_matrix = dist

    def distance_matrix(self) -> np.ndarray:
        """Matriz de distancias dirigidas (BFS). ∞ si no alcanzable."""
        if self._dist_matrix is None:
            self._compute_distance_matrix()
        return self._dist_matrix.copy()

    def shortest_path_length(self, source: Hashable, target: Hashable) -> float:
        """
        g(c_i, c_j): grado de impacto — longitud del camino dirigido más corto.
        (Definición 2.2)
        """
        if self._dist_matrix is None:
            self._compute_distance_matrix()
        d = self._dist_matrix[self._v2idx[source], self._v2idx[target]]
        return int(d) if np.isfinite(d) else float("inf")

    def diameter(self) -> int:
        """Diámetro dirigido: max g(i,j) sobre pares alcanzables."""
        if self._diam is None:
            D = self.distance_matrix()
            finite = D[np.isfinite(D)]
            self._diam = int(np.max(finite)) if len(finite) > 0 else 0
        return self._diam

    # ── Propiedades topológicas (O(1) con pre-cómputo) ───────────────

    def predecessors_direct(self, vertex: Hashable) -> list:
        """Vértices con flechas directas hacia vertex. O(deg)."""
        return list(self._pred_labels[vertex])

    def successors_direct(self, vertex: Hashable) -> list:
        """Vértices alcanzados por flechas directas desde vertex. O(deg)."""
        return list(self._succ_labels[vertex])

    def in_degree(self, vertex: Hashable) -> int:
        """Número de flechas que llegan a vertex. O(1)."""
        return int(self._in_deg[self._v2idx[vertex]])

    def out_degree(self, vertex: Hashable) -> int:
        """Número de flechas que salen de vertex. O(1)."""
        return int(self._out_deg[self._v2idx[vertex]])

    def is_source(self, vertex: Hashable) -> bool:
        """True si no tiene flechas entrantes (candidato a trampa topológica). O(1)."""
        return bool(self._in_deg[self._v2idx[vertex]] == 0)

    def is_sink(self, vertex: Hashable) -> bool:
        """True si no tiene flechas salientes. O(1)."""
        return bool(self._out_deg[self._v2idx[vertex]] == 0)

    def is_acyclic(self) -> bool:
        """True si el quiver no tiene ciclos dirigidos (es un DAG).

        Usa ordenamiento topológico (Kahn) en O(V+E). Cached.
        """
        if self._is_acyclic is not None:
            return self._is_acyclic

        n = self.n_vertices
        in_deg = self._in_deg.copy()
        queue = deque(int(i) for i in range(n) if in_deg[i] == 0)
        visited = 0
        succ = self._succ_idx
        while queue:
            u = queue.popleft()
            visited += 1
            for v in succ[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
        self._is_acyclic = visited == n
        return self._is_acyclic

    # ── Operaciones sobre quivers ────────────────────────────────────

    def opposite(self) -> Quiver:
        """
        Q^op: mismos vértices, flechas invertidas.
        (Proposición 3.2)
        """
        arrows_op = [(f"{name}_op", tgt, src) for name, src, tgt in self._arrows]
        weights_op = {f"{name}_op": self.arrow_weight(name) for name, _, _ in self._arrows}
        return Quiver(self._vertices, arrows_op, weights_op if self._weights else None)

    def symmetrize(self) -> Quiver:
        """
        Q_G: añade flechas inversas para obtener grafo no dirigido simétrico.
        (Proposición 3.1)
        """
        existing = {(src, tgt) for _, src, tgt in self._arrows}
        new_arrows = list(self._arrows)
        counter = 0
        for name, src, tgt in self._arrows:
            if (tgt, src) not in existing:
                new_arrows.append((f"_sym_{counter}", tgt, src))
                existing.add((tgt, src))
                counter += 1
        return Quiver(self._vertices, new_arrows)

    def subquiver(self, vertex_subset: list) -> Quiver:
        """
        Sub-quiver pleno Q|_S: vértices en subset, todas las flechas entre ellos.
        (Ejemplo 3.4)
        """
        vset = set(vertex_subset)
        sub_arrows = [
            (name, src, tgt)
            for name, src, tgt in self._arrows
            if src in vset and tgt in vset
        ]
        sub_weights = {
            a_name: self.arrow_weight(a_name)
            for a_name, _, _ in sub_arrows
            if a_name in self._weights
        }
        return Quiver(vertex_subset, sub_arrows, sub_weights or None)

    def influence_boundary(self, vertex_subset: list, g_max: int) -> set:
        """
        ∂^{g_max} S: vértices fuera de S con caminos de longitud ≤ g_max
        hacia algún vértice de S. (Ejemplo 3.4)
        """
        vset = set(vertex_subset)
        D = self.distance_matrix()
        boundary = set()
        subset_indices = [self._v2idx[c] for c in vertex_subset]
        for j_idx, j in enumerate(self._vertices):
            if j in vset:
                continue
            for c_idx in subset_indices:
                if D[j_idx, c_idx] <= g_max:
                    boundary.add(j)
                    break
        return boundary

    # ── NetworkX ─────────────────────────────────────────────────────

    def to_networkx(self):
        """Convertir a nx.MultiDiGraph."""
        import networkx as nx

        G = nx.MultiDiGraph()
        G.add_nodes_from(self._vertices)
        for name, src, tgt in self._arrows:
            G.add_edge(src, tgt, key=name, weight=self.arrow_weight(name))
        return G

    @classmethod
    def from_networkx(cls, G, weight_attr: str = "weight") -> Quiver:
        """Construir Quiver desde un nx.DiGraph o nx.MultiDiGraph."""
        import networkx as nx

        vertices = list(G.nodes())
        arrows = []
        weights = {}
        counter = 0
        if isinstance(G, nx.MultiDiGraph):
            for u, v, key, data in G.edges(keys=True, data=True):
                name = str(key) if key else f"a_{counter}"
                arrows.append((name, u, v))
                if weight_attr in data:
                    weights[name] = int(data[weight_attr])
                counter += 1
        else:
            for u, v, data in G.edges(data=True):
                name = f"a_{counter}"
                arrows.append((name, u, v))
                if weight_attr in data:
                    weights[name] = int(data[weight_attr])
                counter += 1
        return cls(vertices, arrows, weights or None)

    @classmethod
    def from_adjacency_matrix(cls, A: np.ndarray, labels: Optional[list] = None) -> Quiver:
        """Construir Quiver desde una matriz de adyacencia (entradas enteras)."""
        n = A.shape[0]
        vertices = labels if labels else list(range(n))
        arrows = []
        counter = 0
        for i in range(n):
            for j in range(n):
                for _ in range(int(A[i, j])):
                    arrows.append((f"a_{counter}", vertices[i], vertices[j]))
                    counter += 1
        return cls(vertices, arrows)

    # ── Representación ───────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Quiver(|Q_0|={self.n_vertices}, |Q_1|={self.n_arrows}, "
            f"vertices={self._vertices})"
        )

    def summary(self) -> str:
        """Resumen legible del quiver."""
        lines = [repr(self)]
        lines.append(f"  Flechas: {', '.join(f'{n}:{s}→{t}' for n, s, t in self._arrows)}")
        sources = [v for v in self._vertices if self.is_source(v)]
        sinks = [v for v in self._vertices if self.is_sink(v)]
        if sources:
            lines.append(f"  Fuentes (trampas topológicas): {sources}")
        if sinks:
            lines.append(f"  Sumideros: {sinks}")
        lines.append(f"  DAG: {self.is_acyclic()}")
        lines.append(f"  Diámetro: {self.diameter()}")
        return "\n".join(lines)
