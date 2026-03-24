"""
Brauer configurations and Brauer configuration algebras (BCA).

Theory based on:
  - Green & Schroll (2017): Definition 1.5 — conditions C1-C4.
  - Sierra (2018): center dimension formula.
  - Cañadas, Rodríguez-Nieto, Salazar Díaz (2024): BCA from integer
    partitions, dimension formulas, and branched coverings.

Invariants (Zainea Maya & Moreno Cañadas, 2025):
  δ_B = Σ_{m∈M} μ(m)·val(m)          (Brauer impact factor)
  H(B) = -Σ_{m∈M} p_m · log₂(p_m)    (Brauer entropy)
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Hashable, Optional, Union

from .quiver import Quiver


# ═══════════════════════════════════════════════════════════════════════
# Multiplicity strategies
# ═══════════════════════════════════════════════════════════════════════

def mu_mnti(valency: dict[Hashable, int], **kwargs) -> dict[Hashable, int]:
    """
    Minimum Non-Truncated Integer (MNTI) multiplicity.

    The pointwise smallest μ: Γ₀ → ℕ⁺ such that val(m)·μ(m) > 1
    for every vertex m:
        μ(m) = 2 if val(m) = 1,  μ(m) = 1 if val(m) ≥ 2.

    Under MNTI, ω(m) = μ·val ≥ 2 for all vertices and no vertex is
    truncated (in the sense of Green & Schroll).

    In citation networks, univalent vertices (val=1) are interpreted
    as thematic anchors: specialized references whose depth contribution
    is recognized by the minimum weight ω = 2.
    """
    return {v: 2 if val == 1 else 1 for v, val in valency.items()}


# Backward-compatible alias
mu_standard = mu_mnti


def mu_uniform(valency: dict[Hashable, int], *, value: int = 1, **kwargs) -> dict[Hashable, int]:
    """
    Uniform multiplicity: μ(m) = value for all m.

    Warning: with value=1, vertices with val=1 will have ω=1 and be
    truncated. Condition C3 is only satisfied if every polygon has at
    least one vertex with val ≥ 2.
    """
    return {v: value for v in valency}


def mu_from_data(
    valency: dict[Hashable, int],
    *,
    vertex_data: dict[Hashable, dict],
    field: str = "external_citations",
    scale: Callable[[float], int] = lambda x: max(1, int(math.log2(x + 1)) + 1),
    **kwargs,
) -> dict[Hashable, int]:
    """
    Data-driven multiplicity: μ(m) derived from an external numeric field.

    For citation networks, the field could be the external citation count,
    h-index of the reference's authors, journal impact factor, etc.

    Parameters
    ----------
    vertex_data : dict
        {vertex: {field: numeric_value, ...}}
    field : str
        Key to read from vertex_data[m].
    scale : callable
        Function mapping the raw value to a positive integer μ.
        Default: 1 + floor(log₂(x+1)), giving μ ∈ {1, 2, 3, ...}.
    """
    mu = {}
    for v, val in valency.items():
        raw = vertex_data.get(v, {}).get(field, 0)
        mu[v] = scale(raw) if raw else (2 if val == 1 else 1)
    return mu


MU_STRATEGIES = {
    "mnti": mu_mnti,
    "standard": mu_mnti,      # backward-compatible alias
    "uniform": mu_uniform,
    "from_data": mu_from_data,
}


# ═══════════════════════════════════════════════════════════════════════
# BrauerConfiguration — Configuración de Brauer  M = (M, M₁, μ, O)
# ═══════════════════════════════════════════════════════════════════════

class BrauerConfiguration:
    """
    Brauer configuration Γ = (Γ₀, Γ₁, μ, O).

    Follows Green & Schroll (2017), Definition 1.5, with conditions:
      (C1) Γ₀ is a finite set of vertices. Every vertex belongs to ≥1 polygon.
      (C2) Γ₁ is a finite collection of multisets (polygons) over Γ₀.
           Every polygon has ≥2 vertices.
      (C3) μ: Γ₀ → ℕ⁺ is the multiplicity function. Every polygon has at
           least one vertex m with val(m)·μ(m) > 1.
      (C4) O is an orientation: cyclic ordering at each vertex.

    For citation networks (Zainea Maya & Moreno Cañadas, 2025):
      - Γ₀ = references (cited papers)
      - Γ₁ = articles (each is a multiset of its references)
      - μ  = MNTI multiplicity (thematic specialization depth)
      - O  = chronological order by publication year

    Parameters
    ----------
    vertices : list[Hashable]
        Elements of Γ₀.
    polygons : dict[Hashable, list[Hashable]]
        Polygons (Γ₁). Key = polygon name, value = list of vertices
        (may contain repeats = multiplicity within the multiset).
    mu : dict[Hashable, int] | str | callable | None
        Multiplicity function μ: Γ₀ → ℕ⁺.
        - None, "mnti", or "standard": MNTI multiplicity (μ=2 if val=1, else 1).
        - "uniform": μ=1 for all vertices (may produce truncated vertices).
        - dict: explicit {vertex: int} mapping.
        - callable: function(valency_dict, **kwargs) → {vertex: int}.
    vertex_data : dict[Hashable, dict] | None
        Optional metadata per vertex (e.g. year, title, authors).
    polygon_data : dict[Hashable, dict] | None
        Optional metadata per polygon.
    validate : bool
        If True (default), validates conditions C1, C2, C3 and warns
        on violations. Set to False to skip validation.
    """

    def __init__(
        self,
        vertices: list[Hashable],
        polygons: dict[Hashable, list[Hashable]],
        mu: Optional[Union[dict[Hashable, int], str, Callable]] = None,
        vertex_data: Optional[dict] = None,
        polygon_data: Optional[dict] = None,
        validate: bool = True,
    ):
        self._vertices = list(vertices)
        self._v_set = set(vertices)
        self._polygons = {k: list(v) for k, v in polygons.items()}
        self._vertex_data = vertex_data or {}
        self._polygon_data = polygon_data or {}

        # ── Validate vertices in polygons ──
        for pname, pvertices in self._polygons.items():
            for v in pvertices:
                if v not in self._v_set:
                    raise ValueError(
                        f"Vertex '{v}' of polygon '{pname}' not in Γ₀"
                    )

        # Pre-compute incidence index: vertex → [(polygon_name, multiplicity)]
        self._incidence: dict[Hashable, list[tuple]] = {v: [] for v in self._vertices}
        for pname, pvertices in self._polygons.items():
            counts = Counter(pvertices)
            for v, c in counts.items():
                self._incidence[v].append((pname, c))

        # Compute valencies
        self._valency: dict[Hashable, int] = self._compute_valency()

        # ── Resolve multiplicity ──
        if mu is None or mu == "mnti" or mu == "standard":
            self._mu = mu_mnti(self._valency)
        elif mu == "uniform":
            self._mu = mu_uniform(self._valency)
        elif isinstance(mu, str) and mu in MU_STRATEGIES:
            self._mu = MU_STRATEGIES[mu](
                self._valency, vertex_data=self._vertex_data)
        elif callable(mu):
            self._mu = mu(self._valency, vertex_data=self._vertex_data)
        elif isinstance(mu, dict):
            self._mu = dict(mu)
        else:
            raise ValueError(
                f"mu must be None, a strategy name ({list(MU_STRATEGIES)}), "
                f"a dict, or a callable. Got: {type(mu)}"
            )

        # ── Validate C1, C2, C3 ──
        if validate:
            self._validate_configuration()

        # Caches
        self._successor_seqs: Optional[dict] = None
        self._brauer_quiver: Optional[Quiver] = None
        self._n_loops: Optional[int] = None

    def _validate_configuration(self):
        """Validate conditions C1, C2, C3 from Green & Schroll (2017)."""
        # C1: every vertex belongs to at least one polygon
        for v in self._vertices:
            if not self._incidence[v]:
                warnings.warn(
                    f"C1 violation: vertex '{v}' does not belong to any polygon.",
                    stacklevel=3,
                )

        for pname, pvertices in self._polygons.items():
            # C2: every polygon has at least two vertices
            distinct = len(set(pvertices))
            if distinct < 2:
                warnings.warn(
                    f"C2 violation: polygon '{pname}' has only "
                    f"{distinct} distinct vertex(es).",
                    stacklevel=3,
                )

            # C3: every polygon has at least one vertex with val·μ > 1
            has_nontruncated = False
            for v in set(pvertices):
                if self._valency[v] * self._mu.get(v, 1) > 1:
                    has_nontruncated = True
                    break
            if not has_nontruncated:
                warnings.warn(
                    f"C3 violation: polygon '{pname}' has no vertex m with "
                    f"val(m)·μ(m) > 1. All its vertices are truncated.",
                    stacklevel=3,
                )

    # ── Propiedades básicas ──────────────────────────────────────────

    @property
    def Gamma0(self) -> list:
        """Vértices de la configuración (elementos de M)."""
        return list(self._vertices)

    @property
    def Gamma1(self) -> dict:
        """Polígonos (multiconjuntos). Clave → lista de vértices."""
        return {k: list(v) for k, v in self._polygons.items()}

    @property
    def M(self) -> set:
        """Conjunto base M = ∪ Mᵢ."""
        return set(self._vertices)

    @property
    def n_vertices(self) -> int:
        return len(self._vertices)

    @property
    def n_polygons(self) -> int:
        return len(self._polygons)

    # ── Valencia val(y) — Fórmula (11) del paper ─────────────────────

    def _compute_valency(self) -> dict[Hashable, int]:
        """
        val(y) = Σᵢ fᵢ(y) donde fᵢ(y) es la multiplicidad de y en el
        multiconjunto Mᵢ. (Fórmula 11). Uses pre-computed incidence index.
        """
        return {
            v: sum(c for _, c in self._incidence[v])
            for v in self._vertices
        }

    def valency(self, vertex: Hashable) -> int:
        """val(y): número total de ocurrencias de y en todos los polígonos."""
        return self._valency[vertex]

    # ── Multiplicidad μ ──────────────────────────────────────────────

    def _default_mu(self) -> dict[Hashable, int]:
        """Standard multiplicity (backward compatibility)."""
        return mu_standard(self._valency)

    def mu(self, vertex: Hashable) -> int:
        """Multiplicity μ(m) of vertex m."""
        return self._mu.get(vertex, 1)

    @property
    def mu_dict(self) -> dict[Hashable, int]:
        """Full multiplicity dictionary {vertex: μ(m)}."""
        return dict(self._mu)

    def is_truncated(self, vertex: Hashable) -> bool:
        """A vertex is truncated if val(m)·μ(m) = 1 (Green & Schroll, Def. 1.5).

        Note: under MNTI multiplicity, no vertex is truncated.
        """
        return self._valency[vertex] * self._mu.get(vertex, 1) == 1

    def is_nontruncated(self, vertex: Hashable) -> bool:
        """A vertex is non-truncated if val(m)·μ(m) > 1."""
        return self._valency[vertex] * self._mu[vertex] > 1

    def is_univalent(self, vertex: Hashable) -> bool:
        """A vertex is univalent if val(m) = 1 (independent of μ).

        Univalent vertices generate loops in citation networks and
        receive μ = 2 under the MNTI multiplicity.
        """
        return self._valency[vertex] == 1

    def is_multivalent(self, vertex: Hashable) -> bool:
        """A vertex is multivalent if val(m) ≥ 2 (independent of μ).

        Multivalent vertices bridge multiple polygons and generate
        non-loop arrows in the Brauer quiver.
        """
        return self._valency[vertex] >= 2

    @property
    def univalent_vertices(self) -> list:
        """C_M = {m ∈ Γ₀ : val(m) = 1}."""
        return [v for v in self._vertices if self._valency[v] == 1]

    @property
    def multivalent_vertices(self) -> list:
        """Γ₀ \\ C_M = {m ∈ Γ₀ : val(m) ≥ 2}."""
        return [v for v in self._vertices if self._valency[v] >= 2]

    # ── Conjunto de incidencia I_y ───────────────────────────────────

    def incidence_set(self, vertex: Hashable) -> list[tuple]:
        """
        I_y = {(Mᵢ, fᵢ) ∈ M : y ∈ Mᵢ}
        Retorna lista de (nombre_polígono, multiplicidad_en_polígono)
        para los polígonos que contienen a vertex. O(1) lookup.
        """
        return list(self._incidence[vertex])

    # ── Secuencias de sucesores — Fórmulas (12)-(15) ────────────────

    def successor_sequences(self) -> dict[Hashable, list]:
        """
        Para cada vértice y ∈ M, la secuencia de sucesores S_y es la
        cadena cíclica ordenada de expansiones de los polígonos que contienen
        a y. (Fórmula 15 del paper)

        En redes de citación: los polígonos se ordenan por año de publicación
        (orden cronológico = orientación O).

        Returns
        -------
        dict: vértice → lista de (polígono, copia) en orden de sucesores
        """
        if self._successor_seqs is not None:
            return self._successor_seqs

        seqs = {}
        for v in self._vertices:
            # Obtener polígonos que contienen a v con sus multiplicidades
            incidents = self.incidence_set(v)
            if not incidents:
                seqs[v] = []
                continue

            # Ordenar por la orientación O.
            # Para citaciones: orden por año de publicación del polígono
            incidents_sorted = sorted(
                incidents,
                key=lambda x: self._polygon_sort_key(x[0])
            )

            # Construir la secuencia de sucesores expandida
            # Cada (Mᵢ, fᵢ) se expande en fᵢ copias: Mᵢ^(1), ..., Mᵢ^(fᵢ)
            seq = []
            for pname, fi in incidents_sorted:
                for copy in range(1, fi + 1):
                    seq.append((pname, copy))

            seqs[v] = seq

        self._successor_seqs = seqs
        return seqs

    def _polygon_sort_key(self, polygon_name: Hashable):
        """
        Clave de ordenamiento para polígonos.
        Usa el año de publicación si está disponible, sino el nombre.
        """
        pdata = self._polygon_data.get(polygon_name, {})
        return (pdata.get("year", 0), str(polygon_name))

    def compressed_successor_sequence(self, vertex: Hashable) -> list:
        """
        Secuencia de sucesores comprimida: solo los polígonos (sin copias).
        M₁^(1) < M₂^(1) < ... < M_{|I_y|}^(1)
        (Fórmula 15, primera línea)
        """
        seqs = self.successor_sequences()
        return [pname for pname, _ in seqs.get(vertex, [])]

    # ── Quiver de Brauer Q_M — Sección Brauer Configuration Algebras ──

    def brauer_quiver(self) -> Quiver:
        """
        Construir el quiver de Brauer Q_M inducido por la configuración.

        Q₀ = M₁ (los polígonos son los vértices del quiver)
        Q₁: para cada par {(Mᵢ,fᵢ), Suc((Mᵢ,fᵢ))} en la secuencia
             de sucesores de un vértice y con val(y) > 1, hay una flecha
             de (Mᵢ,fᵢ) a Suc((Mᵢ,fᵢ)).

        (Sección "Brauer Configuration Algebras" del paper, después de
        Proposition 3)
        """
        if self._brauer_quiver is not None:
            return self._brauer_quiver

        polygon_names = list(self._polygons.keys())
        arrows = []
        n_loops = 0

        seqs = self.successor_sequences()
        for v in self._vertices:
            seq = seqs.get(v, [])
            if not seq:
                continue

            # Flechas entre sucesivos en la secuencia (cíclica).
            # Para val(y) = 1 la secuencia tiene 1 elemento y el ciclo
            # Suc(M^(1)) = M^(1) genera un self-loop.
            for i in range(len(seq)):
                src_poly = seq[i][0]
                tgt_poly = seq[(i + 1) % len(seq)][0]
                arrow_name = f"s_{v}_{src_poly}_to_{tgt_poly}_{i}"
                arrows.append((arrow_name, src_poly, tgt_poly))
                if src_poly == tgt_poly:
                    n_loops += 1

        self._brauer_quiver = Quiver(polygon_names, arrows)
        self._n_loops = n_loops
        return self._brauer_quiver

    # ── Grafo de cubrimiento c(Q_M) — Fórmula (21) ──────────────────

    def covering_graph_edges(self) -> list[tuple]:
        """
        Aristas del grafo de cubrimiento c(Q_M).
        {Mᵢ, Mⱼ} ∈ E si y solo si existe un vértice y con μ(y) = 1
        tal que Mᵢ^(fᵢ(y)) < Mⱼ^(1) en alguna secuencia de sucesores.
        (Fórmula 21)
        """
        edges = set()
        seqs = self.successor_sequences()
        for v in self._vertices:
            if self._mu.get(v, 1) != 1:
                continue
            seq = seqs.get(v, [])
            for i in range(len(seq) - 1):
                p1 = seq[i][0]
                p2 = seq[i + 1][0]
                if p1 != p2:
                    edge = tuple(sorted([str(p1), str(p2)]))
                    edges.add(edge)
        return list(edges)

    # ── Dimensión del álgebra Λ_M — Fórmula (22) ────────────────────

    def dimension(self) -> int:
        """
        dim_k Λ_M = 2|M₁| + Σ_{mᵢ∈M} val(mᵢ)(μ(mᵢ)·val(mᵢ) - 1)
        (Fórmula 22 del paper, Remark 2 (B₆))
        """
        n_poly = self.n_polygons
        total = 2 * n_poly
        for v in self._vertices:
            val_v = self._valency[v]
            mu_v = self._mu[v]
            total += val_v * (mu_v * val_v - 1)
        return total

    # ── Loops del quiver de Brauer ───────────────────────────────────

    def n_loops(self) -> int:
        """
        Número de loops en Q_M.
        Un loop ocurre en el polígono Mᵢ cuando un vértice y tiene
        multiplicidad fᵢ(y) > 1 dentro de Mᵢ, generando flechas internas.
        Cached during brauer_quiver() construction.
        """
        if self._n_loops is None:
            self.brauer_quiver()  # triggers computation and caches _n_loops
        return self._n_loops

    # ── Dimensión del centro Z(Λ_M) — Fórmula (23) ──────────────────

    def center_dimension(self) -> int:
        """
        dim_k Z(Λ_M) = 1 + Σ_{m∈M} μ(m) + |M₁| - |M| + #Loops(Q_M) - |C_M|
        donde C_M = {m ∈ M | val(m) = 1} y μ(m) > 1.
        (Fórmula 23 del paper, Remark 2 (B₇))
        """
        sum_mu = sum(self._mu[v] for v in self._vertices)
        n_poly = self.n_polygons
        n_vert = self.n_vertices
        loops = self.n_loops()

        # C_M: vértices con val = 1 y μ > 1 (los truncados con μ forzado)
        c_m = sum(
            1 for v in self._vertices
            if self._valency[v] == 1 and self._mu[v] > 1
        )

        return 1 + sum_mu + n_poly - n_vert + loops - c_m

    # ── Factor de impacto δ_B — Fórmula manuscrita ──────────────────

    def impact_factor(self) -> float:
        """
        Factor de impacto de la configuración de Brauer.
        δ_B = Σ_{m∈M} μ(m)·val(m)
        (Fórmula manuscrita de Agustín Moreno Cañadas)
        """
        return sum(
            self._mu[v] * self._valency[v]
            for v in self._vertices
        )

    # ── Entropía H(B) — Fórmula manuscrita ───────────────────────────

    def entropy(self) -> float:
        """
        Entropía de la configuración de Brauer.
        H(B) = -Σ_{m∈M} [μ(m)·val(m)/δ_B] · log₂[μ(m)·val(m)/δ_B]
        (Fórmula manuscrita de Agustín Moreno Cañadas)

        Análoga a la entropía de Shannon, mide la distribución de
        influencia en la red de citación.
        """
        delta = self.impact_factor()
        if delta == 0:
            return 0.0

        h = 0.0
        for v in self._vertices:
            weight = self._mu[v] * self._valency[v]
            if weight > 0:
                p = weight / delta
                h -= p * math.log2(p)
        return h

    # ── Palabra w(M) del multiconjunto — Fórmula (10) ────────────────

    def word(self, polygon_name: Hashable) -> str:
        """
        Palabra w(M) del multiconjunto M (polígono).
        w(M) = m₁^{f(m₁)} m₂^{f(m₂)} ... mₛ^{f(mₛ)}
        (Fórmula 10)
        """
        pvertices = self._polygons[polygon_name]
        counts = Counter(pvertices)
        parts = []
        for v, c in counts.items():
            label = str(v)
            if c > 1:
                parts.append(f"{label}^{{{c}}}")
            else:
                parts.append(label)
        return " · ".join(parts)

    # ── Mensaje de Brauer M(M) — Fórmula (20) ───────────────────────

    def brauer_message(self) -> str:
        """
        M(M) = w(M₁)w(M₂)...w(Mₕ)
        Concatenación de las palabras fijas de todos los polígonos.
        (Fórmula 20)
        """
        words = [self.word(p) for p in self._polygons]
        return " | ".join(words)

    # ── Defecto ν(D) — para cubrimientos ramificados ────────────────

    def defect(self) -> int:
        """
        Defecto total ν(D) = Σ ν(Lⱼ) donde ν(L) = Σ(λᵢ - 1).
        Para cada polígono, el defecto es |polígono| - #{partes distintas}.
        (Sección 2.3 del paper)
        """
        total = 0
        for pvertices in self._polygons.values():
            counts = Counter(pvertices)
            # ν(L) = |L| - #{partes} = Σ(fᵢ - 1)
            for c in counts.values():
                total += c - 1
        return total

    # ── Análisis de Brauer completo ──────────────────────────────────

    def brauer_analysis(self) -> dict:
        """
        Análisis de Brauer completo de la configuración.
        Retorna un diccionario con todos los invariantes algebraicos.
        (Tabla 2 y Tabla 3 del paper)
        """
        return {
            "n_vertices": self.n_vertices,
            "n_polygons": self.n_polygons,
            "dimension": self.dimension(),
            "center_dimension": self.center_dimension(),
            "n_loops": self.n_loops(),
            "defect": self.defect(),
            "impact_factor_delta_B": self.impact_factor(),
            "entropy_H_B": self.entropy(),
            "valencies": {str(v): self._valency[v] for v in self._vertices},
            "multiplicities": {str(v): self._mu[v] for v in self._vertices},
            "truncated_vertices": [
                str(v) for v in self._vertices if self.is_truncated(v)
            ],
            "univalent_vertices": [
                str(v) for v in self._vertices if self.is_univalent(v)
            ],
            "multivalent_vertices": [
                str(v) for v in self._vertices if self.is_multivalent(v)
            ],
        }

    # ── Representación ───────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BrauerConfiguration(|Γ₀|={self.n_vertices}, "
            f"|Γ₁|={self.n_polygons}, "
            f"δ_B={self.impact_factor()}, "
            f"H(B)={self.entropy():.4f})"
        )

    def summary(self) -> str:
        """Resumen legible de la configuración."""
        analysis = self.brauer_analysis()
        lines = [repr(self)]
        lines.append(f"  dim_k Λ_M = {analysis['dimension']}")
        lines.append(f"  dim_k Z(Λ_M) = {analysis['center_dimension']}")
        lines.append(f"  #Loops(Q_M) = {analysis['n_loops']}")
        lines.append(f"  ν(D) = {analysis['defect']}")
        lines.append(f"  δ_B = {analysis['impact_factor_delta_B']}")
        lines.append(f"  H(B) = {analysis['entropy_H_B']:.6f}")
        lines.append(
            f"  Univalentes: {len(analysis['univalent_vertices'])}, "
            f"Multivalentes: {len(analysis['multivalent_vertices'])}"
        )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Construcción desde datos de citación JSON
# ═══════════════════════════════════════════════════════════════════════

def brauer_from_citation_json(json_path: str | Path) -> BrauerConfiguration:
    """
    Construir una configuración de Brauer desde un archivo JSON de citación.

    El JSON debe tener la estructura:
      {
        "papers": [{id, year, references: [ref_id, ...]}, ...],
        "reference_pool": {ref_id: {year, title, ...}, ...}
      }

    La configuración resultante:
      - Γ₀ = reference_pool (vértices = referencias)
      - Γ₁ = papers (polígonos = artículos, cada uno es multiconjunto de refs)
      - O = orden por año de publicación (instrucción de Agustín)

    Parameters
    ----------
    json_path : str | Path
        Ruta al archivo JSON.

    Returns
    -------
    BrauerConfiguration
    """
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data["papers"]
    ref_pool = data["reference_pool"]

    # Γ₀: todos los IDs de referencias que aparecen en algún paper
    all_refs = set()
    for paper in papers:
        for ref_id in paper.get("references", []):
            all_refs.add(ref_id)

    vertices = sorted(all_refs)

    # Vertex data: metadatos de cada referencia
    vertex_data = {}
    for ref_id in vertices:
        if ref_id in ref_pool:
            vertex_data[ref_id] = ref_pool[ref_id]
        else:
            # Podría ser una auto-referencia a un paper principal
            for p in papers:
                if p["id"] == ref_id:
                    vertex_data[ref_id] = {
                        "year": p.get("year", 0),
                        "title": p.get("title", ""),
                        "authors": p.get("authors", []),
                    }
                    break

    # Γ₁: polígonos = papers, cada uno con sus refs ordenadas por año
    polygons = {}
    polygon_data = {}
    for paper in papers:
        pid = paper["id"]
        refs = paper.get("references", [])

        # Ordenar por año de publicación (instrucción de Agustín)
        refs_sorted = sorted(
            refs,
            key=lambda r: (
                vertex_data.get(r, {}).get("year", 0),
                str(r),
            ),
        )

        polygons[pid] = refs_sorted
        polygon_data[pid] = {
            "year": paper.get("year", 0),
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "journal": paper.get("journal", ""),
            "doi": paper.get("doi", ""),
        }

    return BrauerConfiguration(
        vertices=vertices,
        polygons=polygons,
        vertex_data=vertex_data,
        polygon_data=polygon_data,
    )


# ═══════════════════════════════════════════════════════════════════════
# Ejemplos del paper (mathematics-12-03626-v2)
# ═══════════════════════════════════════════════════════════════════════

def example_partitions_of_10() -> BrauerConfiguration:
    """
    Ejemplo 2 del paper: particiones de 10 como branch data sobre S².

    L₁ = (4^{2} 2^{1}),  L₂ = (3^{3} 1^{1}),  L₃ = (2^{5})
    M = {1, 2, 3, 4}

    Secuencias de sucesores (identidades 17):
      S₁ = L₂^{1}  (solo aparece en L₂)
      S₂ = L₁^{1} < L₃^{1} < L₃^{2} < L₃^{3} < L₃^{4} < L₃^{5}
      S₃ = L₂^{1} < L₂^{2} < L₂^{3}
      S₄ = L₁^{1} < L₁^{2}

    val(1)=1, val(2)=6, val(3)=3, val(4)=2
    μ(1)=2, μ(i)=1 para i∈{2,3,4}
    """
    vertices = [1, 2, 3, 4]
    polygons = {
        "L1": [4, 4, 2],        # (4^2, 2^1)
        "L2": [3, 3, 3, 1],     # (3^3, 1^1)
        "L3": [2, 2, 2, 2, 2],  # (2^5)
    }
    mu = {1: 2, 2: 1, 3: 1, 4: 1}

    return BrauerConfiguration(vertices, polygons, mu=mu)


def example_compositions_B7() -> BrauerConfiguration:
    """
    Ejemplo 7 del paper: composiciones de tipo B(j,m;7).

    L₁ = {4,2,1}, L₂ = {4,1,1,1}, L₃ = {2,2,2,1},
    L₄ = {2,2,1,1,1}, L₅ = {2,1,1,1,1,1}, L₆ = {1,1,1,1,1,1,1}

    M = {1,2,4}, M₁ = {L₁,L₂,L₃,L₄}
    val(1) = 8, val(2) = 6, val(4) = 2
    """
    vertices = [1, 2, 4]
    polygons = {
        "L1": [4, 2, 1],
        "L2": [4, 1, 1, 1],
        "L3": [2, 2, 2, 1],
        "L4": [2, 2, 1, 1, 1],
    }

    return BrauerConfiguration(vertices, polygons)
