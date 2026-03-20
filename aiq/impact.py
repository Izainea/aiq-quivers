"""
Métricas de impacto: grado, vector, SFV y tasas de impacto.

Referencia: ACT.tex, Definiciones 2.2 (grado de impacto), 2.5 (vector de impacto),
2.3 (SFV), 2.9 (tasa de impacto), 2.11 (AIQ con signo).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .quiver import Quiver


# ═══════════════════════════════════════════════════════════════════════
# Grado y vector de impacto
# ═══════════════════════════════════════════════════════════════════════

def impact_degree(quiver: Quiver, source, target) -> float:
    """
    g(c_i, c_j): grado de impacto — longitud del camino dirigido más corto.
    Retorna inf si no es alcanzable. (Definición 2.2)
    """
    return quiver.shortest_path_length(source, target)


def impact_vector(quiver: Quiver, source, target,
                  max_k: Optional[int] = None) -> np.ndarray:
    """
    vec_g(c_i, c_j) = (n_0, n_1, ..., n_{max_k}) donde n_k = (A^k)_{ij}.
    (Definición 2.5, Proposición 2.5)
    """
    if max_k is None:
        max_k = quiver.diameter() if quiver.is_acyclic() else 4
    i = quiver.vertex_index(source)
    j = quiver.vertex_index(target)
    vec = np.zeros(max_k + 1, dtype=np.float64)
    for k in range(max_k + 1):
        Ak = quiver.adjacency_power(k)
        vec[k] = Ak[i, j]
    return vec


def impact_vector_matrix(quiver: Quiver,
                         max_k: Optional[int] = None) -> np.ndarray:
    """
    Todos los vectores de impacto simultáneamente.
    Retorna array V de forma (n, n, max_k+1) donde V[i,j,k] = (A^k)_{ij}.
    """
    if max_k is None:
        max_k = quiver.diameter() if quiver.is_acyclic() else 4
    n = quiver.n_vertices
    V = np.zeros((n, n, max_k + 1), dtype=np.float64)
    for k in range(max_k + 1):
        V[:, :, k] = quiver.adjacency_power(k)
    return V


# ═══════════════════════════════════════════════════════════════════════
# Sistema Fundamental de Vecindades (SFV)
# ═══════════════════════════════════════════════════════════════════════

class FundamentalNeighborhoodSystem:
    """
    SFV de una célula c en el quiver Q.
    (Definición 2.3, Teorema 2.1)

    Computa la familia anidada {A_k(c)} y las capas N_g(c).
    Por defecto usa vecindades de entrada (predecesores): las células
    que pueden enviar influencia hacia c.
    """

    __slots__ = ("quiver", "cell", "direction", "g_max", "_layers", "_balls")

    def __init__(self, quiver: Quiver, cell, g_max: Optional[int] = None,
                 direction: str = "in", _dist_matrix: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        direction : 'in' o 'out'
            'in': predecesores (para reglas de evolución, convención Def. 2.3)
            'out': sucesores
        _dist_matrix : np.ndarray or None
            Pre-computed distance matrix to avoid recomputation.
        """
        self.quiver = quiver
        self.cell = cell
        self.direction = direction

        if g_max is None:
            g_max = quiver.diameter()
        self.g_max = g_max

        self._layers: dict[int, set] = {}
        self._balls: dict[int, set] = {}
        self._compute(_dist_matrix)

    def _compute(self, dist_matrix: Optional[np.ndarray] = None):
        """BFS en la dirección apropiada para calcular distancias."""
        if dist_matrix is None:
            D = self.quiver.distance_matrix()
        else:
            D = dist_matrix

        c_idx = self.quiver.vertex_index(self.cell)
        verts = self.quiver.Q0

        # Pre-extract the column/row we need
        if self.direction == "in":
            distances = D[:, c_idx]  # dist(j → c) for all j
        else:
            distances = D[c_idx, :]  # dist(c → j) for all j

        for g in range(self.g_max + 1):
            layer = set()
            ball = set()
            for j_idx, v in enumerate(verts):
                d = distances[j_idx]
                if d <= g:
                    ball.add(v)
                if abs(d - g) < 0.5 and np.isfinite(d):
                    layer.add(v)
            self._balls[g] = ball
            self._layers[g] = layer

    def A(self, k: int) -> set:
        """
        A_k(c): bola de radio k — vértices a distancia ≤ k.
        (Definición 2.3)
        """
        if k > self.g_max:
            return self._balls.get(self.g_max, set())
        return self._balls.get(k, set())

    def layer(self, g: int) -> set:
        """
        N_g(c) = A_g(c) \\ A_{g-1}(c): vértices a distancia exactamente g.
        N_0(c) = {c}. (Teorema 2.1, partición en capas disjuntas)
        """
        return self._layers.get(g, set())

    def Delta(self, g: int) -> int:
        """|N_g(c)|: número de células en la capa g."""
        return len(self.layer(g))

    def all_layers(self) -> dict[int, set]:
        """Retorna {g: conjunto_de_vértices} para g = 0, ..., g_max."""
        return {g: self.layer(g) for g in range(self.g_max + 1)}

    def is_topologically_trapped(self) -> bool:
        """
        True si todas las capas g ≥ 1 están vacías: la célula es una fuente
        y no puede recibir influencia. (Ejemplo 3.1, trampa topológica)
        """
        for g in range(1, self.g_max + 1):
            if self.Delta(g) > 0:
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════
# Pre-computed layer cache for batch rate computation
# ═══════════════════════════════════════════════════════════════════════

class _LayerCache:
    """Batch FNS computation: computes all layers for all vertices at once."""

    __slots__ = ("_layers",)

    def __init__(self, quiver: Quiver, g_max: int, direction: str = "in"):
        n = quiver.n_vertices
        verts = quiver.Q0
        D = quiver.distance_matrix()

        # layers[cell_idx][g] = list of (vertex_label, vertex_idx)
        self._layers: list[list[list[tuple]]] = [
            [[] for _ in range(g_max + 1)] for _ in range(n)
        ]

        for c_idx in range(n):
            if direction == "in":
                distances = D[:, c_idx]
            else:
                distances = D[c_idx, :]

            for j_idx in range(n):
                d = distances[j_idx]
                if np.isfinite(d):
                    g = int(d + 0.5)
                    if 0 <= g <= g_max:
                        self._layers[c_idx][g].append((verts[j_idx], j_idx))

    def layer(self, cell_idx: int, g: int) -> list[tuple]:
        """Returns list of (vertex_label, vertex_idx) at distance g from cell."""
        try:
            return self._layers[cell_idx][g]
        except IndexError:
            return []


# ═══════════════════════════════════════════════════════════════════════
# Tasas de impacto
# ═══════════════════════════════════════════════════════════════════════

def impact_rate_simple(
    quiver: Quiver,
    cell,
    config: dict,
    target_state,
    beta: float,
    alpha: float,
    P: Callable[[int], float],
    g_max: Optional[int] = None,
    _layer_cache: Optional[_LayerCache] = None,
) -> float:
    """
    Tasa de impacto simple i^t(c).
    (Definición 2.9)

    i^t(c) = (β/α) Σ_{g=1}^{g_max} (σ_{g,K}^t(c) / Δ_g(c)) · P(g)
    """
    if g_max is None:
        g_max = quiver.diameter()

    if _layer_cache is not None:
        c_idx = quiver.vertex_index(cell)
        rate = 0.0
        for g in range(1, g_max + 1):
            layer = _layer_cache.layer(c_idx, g)
            delta_g = len(layer)
            if delta_g == 0:
                continue
            sigma_g = sum(1 for v, _ in layer if config.get(v) == target_state)
            rate += (sigma_g / delta_g) * P(g)
        return (beta / alpha) * rate

    fns = FundamentalNeighborhoodSystem(quiver, cell, g_max, direction="in")
    rate = 0.0

    for g in range(1, g_max + 1):
        layer = fns.layer(g)
        delta_g = len(layer)
        if delta_g == 0:
            continue
        sigma_g = sum(1 for v in layer if config.get(v) == target_state)
        rate += (sigma_g / delta_g) * P(g)

    return (beta / alpha) * rate


def impact_rate_enriched(
    quiver: Quiver,
    cell,
    config: dict,
    target_state,
    beta: float,
    alpha: float,
    P: Callable[[int], float],
    g_max: Optional[int] = None,
    quotient=None,
) -> float:
    """
    Tasa de impacto enriquecida ĩ^t(c).
    (Sección 2.4, fórmula después de Definición 2.10)

    ĩ^t(c) = (β/α) Σ_{g=1}^{g_max} (σ̃_{g,K}/Δ̃_g) · P(g)

    donde σ̃_{g,K}(c) = Σ_{j: π(j)=K} (A^g)_{jc}
          Δ̃_g(c) = Σ_j (A^g)_{jc}
    """
    if g_max is None:
        g_max = quiver.diameter()

    c_idx = quiver.vertex_index(cell)
    verts = quiver.Q0
    rate = 0.0

    for g in range(1, g_max + 1):
        if quotient is not None:
            sigma_tilde = 0.0
            delta_tilde = 0.0
            for v in verts:
                dim_jc = quotient.dimension(v, cell, g)
                delta_tilde += dim_jc
                if config.get(v) == target_state:
                    sigma_tilde += dim_jc
        else:
            Ag = quiver.adjacency_power(g)
            col = Ag[:, c_idx]
            delta_tilde = col.sum()
            sigma_tilde = sum(
                col[quiver.vertex_index(v)]
                for v in verts
                if config.get(v) == target_state
            )

        if delta_tilde < 1e-12:
            continue
        rate += (sigma_tilde / delta_tilde) * P(g)

    return (beta / alpha) * rate


def impact_rate_signed(
    quiver: Quiver,
    cell,
    config: dict,
    target_state,
    beta: float,
    alpha: float,
    P: Callable[[int], float],
    g_max: Optional[int] = None,
) -> float:
    """
    Tasa de impacto enriquecida con signo ĩ^t_±(c).
    (Definición 2.11)

    Usa W^g (matriz de adyacencia con signo) para los walks con signo.
    La probabilidad efectiva es min(max(tasa, 0), 1).
    """
    if g_max is None:
        g_max = quiver.diameter()

    c_idx = quiver.vertex_index(cell)
    verts = quiver.Q0

    rate = 0.0
    for g in range(1, g_max + 1):
        Ag = quiver.adjacency_power(g)
        Wg = quiver.signed_adjacency_power(g)

        col_A = Ag[:, c_idx]
        col_W = Wg[:, c_idx]

        delta_tilde = col_A.sum()
        if delta_tilde < 1e-12:
            continue

        sigma_pm = sum(
            col_W[quiver.vertex_index(v)]
            for v in verts
            if config.get(v) == target_state
        )
        rate += (sigma_pm / delta_tilde) * P(g)

    return (beta / alpha) * rate


def effective_transition_probability(rate: float) -> float:
    """
    min(max(tasa, 0), 1): probabilidad efectiva de transición.
    (Observación 2.2 y Definición 2.11)
    """
    return min(max(rate, 0.0), 1.0)
