"""
Álgebra de Caminos kQ, ideales y cociente kQ/I.

Referencia: ACT.tex, Definición 2.6 (Álgebra de Caminos), Definición 2.8 (Álgebra
con Relaciones), Proposición 2.2 (invariante algebraico).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from .quiver import Quiver


# ═══════════════════════════════════════════════════════════════════════
# Path — Un walk dirigido en el quiver
# ═══════════════════════════════════════════════════════════════════════

class Path:
    """
    Un walk dirigido p = α₁α₂⋯αₙ en un quiver (secuencia de flechas
    concatenables). Los caminos triviales e_i tienen longitud 0.

    Referencia: Definición 2.6 y Observación 2.2 (walks vs caminos simples).
    """

    def __init__(self, quiver: Quiver, *, arrows: Optional[list[str]] = None,
                 vertex=None):
        """
        Crear un camino.
        - arrows: lista de nombres de flechas formando un walk válido.
        - vertex: etiqueta de vértice (camino trivial e_i).
        Exactamente uno de los dos debe proporcionarse.
        """
        self.quiver = quiver
        if vertex is not None and arrows is not None:
            raise ValueError("Proporcionar arrows o vertex, no ambos")
        if vertex is not None:
            if vertex not in quiver._v2idx:
                raise ValueError(f"Vértice '{vertex}' no está en Q_0")
            self._arrows = []
            self._source = vertex
            self._target = vertex
        elif arrows is not None and len(arrows) == 0:
            raise ValueError("Lista de flechas vacía; use vertex= para caminos triviales")
        elif arrows is not None:
            self._arrows = list(arrows)
            self._source = quiver.source(arrows[0])
            self._target = quiver.target(arrows[-1])
            # Validar concatenabilidad
            for i in range(len(arrows) - 1):
                if quiver.target(arrows[i]) != quiver.source(arrows[i + 1]):
                    raise ValueError(
                        f"Flechas no concatenables: t({arrows[i]})="
                        f"{quiver.target(arrows[i])} ≠ s({arrows[i+1]})="
                        f"{quiver.source(arrows[i+1])}"
                    )
        else:
            raise ValueError("Debe proporcionar arrows o vertex")

    @property
    def length(self) -> int:
        return len(self._arrows)

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @property
    def arrows(self) -> list[str]:
        return list(self._arrows)

    def concatenate(self, other: Path) -> Optional[Path]:
        """
        Concatenación p·q. Retorna None (= 0 en kQ) si t(p) ≠ s(q).
        (Definición 2.6, regla de multiplicación)
        """
        if self._target != other._source:
            return None
        if self.length == 0:
            return other
        if other.length == 0:
            return self
        return Path(self.quiver, arrows=self._arrows + other._arrows)

    def sign(self) -> int:
        """
        Producto de pesos w(αᵢ) para quivers con signo.
        (Definición 2.11: extensión multiplicativa a walks)
        """
        s = 1
        for a in self._arrows:
            s *= self.quiver.arrow_weight(a)
        return s

    def __eq__(self, other) -> bool:
        if not isinstance(other, Path):
            return False
        return (self._source == other._source and
                self._target == other._target and
                self._arrows == other._arrows)

    def __hash__(self) -> int:
        return hash((self._source, tuple(self._arrows), self._target))

    def __repr__(self) -> str:
        if self.length == 0:
            return f"e_{self._source}"
        return " · ".join(self._arrows)


# ═══════════════════════════════════════════════════════════════════════
# PathAlgebraElement — Combinación lineal de caminos
# ═══════════════════════════════════════════════════════════════════════

class PathAlgebraElement:
    """
    Elemento de kQ: combinación lineal formal Σ aᵢ pᵢ de caminos.

    Soporta +, -, multiplicación por concatenación (extendida linealmente),
    y multiplicación escalar.
    """

    def __init__(self, terms: dict[Path, float]):
        self.terms = {p: c for p, c in terms.items() if abs(c) > 1e-12}

    def __add__(self, other: PathAlgebraElement) -> PathAlgebraElement:
        result = dict(self.terms)
        for p, c in other.terms.items():
            result[p] = result.get(p, 0.0) + c
        return PathAlgebraElement(result)

    def __sub__(self, other: PathAlgebraElement) -> PathAlgebraElement:
        result = dict(self.terms)
        for p, c in other.terms.items():
            result[p] = result.get(p, 0.0) - c
        return PathAlgebraElement(result)

    def __mul__(self, other: PathAlgebraElement) -> PathAlgebraElement:
        """Producto por concatenación extendido linealmente."""
        result: dict[Path, float] = {}
        for p1, c1 in self.terms.items():
            for p2, c2 in other.terms.items():
                prod = p1.concatenate(p2)
                if prod is not None:
                    result[prod] = result.get(prod, 0.0) + c1 * c2
        return PathAlgebraElement(result)

    def __rmul__(self, scalar: float) -> PathAlgebraElement:
        return PathAlgebraElement({p: scalar * c for p, c in self.terms.items()})

    def __neg__(self) -> PathAlgebraElement:
        return PathAlgebraElement({p: -c for p, c in self.terms.items()})

    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def is_homogeneous(self) -> Optional[int]:
        """Retorna el grado si es homogéneo, None si no lo es."""
        if not self.terms:
            return 0
        degrees = {p.length for p in self.terms}
        return degrees.pop() if len(degrees) == 1 else None

    def grade_decomposition(self) -> dict[int, PathAlgebraElement]:
        """Descomposición en componentes graduadas kQ = ⊕ kQ_n."""
        graded: dict[int, dict[Path, float]] = defaultdict(dict)
        for p, c in self.terms.items():
            graded[p.length][p] = c
        return {k: PathAlgebraElement(v) for k, v in graded.items()}

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for p, c in self.terms.items():
            if abs(c - 1.0) < 1e-12:
                parts.append(str(p))
            elif abs(c + 1.0) < 1e-12:
                parts.append(f"-{p}")
            else:
                parts.append(f"{c}·{p}")
        return " + ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# PathAlgebra — El álgebra de caminos kQ
# ═══════════════════════════════════════════════════════════════════════

class PathAlgebra:
    """
    Álgebra de caminos kQ de un quiver Q.

    Permite enumerar caminos, calcular dimensiones de espacios e_i·kQ_k·e_j,
    y verificar la equivalencia con potencias de la matriz de adyacencia
    (Proposición 2.5).
    """

    def __init__(self, quiver: Quiver):
        self.quiver = quiver
        self._paths_cache: dict[int, list[Path]] = {}
        # Construir lookup: vértice → flechas que salen de él
        self._outgoing: dict = defaultdict(list)
        for name, src, _ in quiver.Q1:
            self._outgoing[src].append(name)

    def enumerate_paths(self, length: int) -> list[Path]:
        """Enumerar todos los walks de longitud dada."""
        if length in self._paths_cache:
            return self._paths_cache[length]

        if length == 0:
            paths = [Path(self.quiver, vertex=v) for v in self.quiver.Q0]
        elif length == 1:
            paths = [Path(self.quiver, arrows=[name]) for name, _, _ in self.quiver.Q1]
        else:
            prev = self.enumerate_paths(length - 1)
            paths = []
            for p in prev:
                for arrow_name in self._outgoing.get(p.target, []):
                    new_path = Path(self.quiver, arrows=p.arrows + [arrow_name])
                    paths.append(new_path)
        self._paths_cache[length] = paths
        return paths

    def paths_from_to(self, source, target, length: int) -> list[Path]:
        """
        Caminos de longitud dada de source a target.
        Base de e_{source} · kQ_{length} · e_{target}.
        """
        return [
            p for p in self.enumerate_paths(length)
            if p.source == source and p.target == target
        ]

    def dimension(self, source, target, length: int) -> int:
        """
        dim(e_{source} · kQ_{length} · e_{target}).
        Debe ser igual a (A^length)_{source, target}. (Proposición 2.2)
        """
        return len(self.paths_from_to(source, target, length))

    def dimension_via_matrix(self, source, target, length: int) -> int:
        """Calcular la dimensión usando A^k (método matricial eficiente)."""
        Ak = self.quiver.adjacency_power(length)
        i = self.quiver.vertex_index(source)
        j = self.quiver.vertex_index(target)
        return int(Ak[i, j])

    def graded_dimension(self, length: int) -> int:
        """dim(kQ_n): número total de walks de longitud dada."""
        return len(self.enumerate_paths(length))

    def total_dimension(self, max_length: Optional[int] = None) -> int:
        """dim(kQ) truncada a max_length."""
        if max_length is None:
            if self.quiver.is_acyclic():
                max_length = self.quiver.diameter()
            else:
                raise ValueError("Quiver tiene ciclos: especificar max_length")
        return sum(self.graded_dimension(k) for k in range(max_length + 1))

    def idempotent(self, vertex) -> PathAlgebraElement:
        """Idempotente e_v."""
        return PathAlgebraElement({Path(self.quiver, vertex=vertex): 1.0})

    def arrow_element(self, arrow_name: str) -> PathAlgebraElement:
        """Elemento correspondiente a una flecha."""
        return PathAlgebraElement({Path(self.quiver, arrows=[arrow_name]): 1.0})

    def path_element(self, arrows: list[str]) -> PathAlgebraElement:
        """Elemento correspondiente a un camino dado."""
        return PathAlgebraElement({Path(self.quiver, arrows=arrows): 1.0})

    def verify_matrix_equivalence(self, max_k: Optional[int] = None) -> bool:
        """
        Verificar Proposición 2.5: para todo i, j, k,
        dim(e_i · kQ_k · e_j) == (A^k)_{ij}.
        """
        if max_k is None:
            max_k = self.quiver.diameter() if self.quiver.is_acyclic() else 4
        for k in range(max_k + 1):
            Ak = self.quiver.adjacency_power(k)
            for i, vi in enumerate(self.quiver.Q0):
                for j, vj in enumerate(self.quiver.Q0):
                    dim_alg = self.dimension(vi, vj, k)
                    dim_mat = int(Ak[i, j])
                    if dim_alg != dim_mat:
                        print(
                            f"FALLO: dim(e_{vi}·kQ_{k}·e_{vj}) = {dim_alg} "
                            f"≠ (A^{k})_{{{i},{j}}} = {dim_mat}"
                        )
                        return False
        return True


# ═══════════════════════════════════════════════════════════════════════
# Ideal — Ideal bilateral de kQ
# ═══════════════════════════════════════════════════════════════════════

class Ideal:
    """
    Ideal I de kQ definido por una lista de generadores.

    Cada generador es un PathAlgebraElement homogéneo (p.ej. α₁β − α₂β).
    (Definición 2.8)
    """

    def __init__(self, algebra: PathAlgebra, generators: list[PathAlgebraElement]):
        self.algebra = algebra
        self.generators = generators
        self._build_reduction_rules()

    def _build_reduction_rules(self):
        """
        Construir reglas de reducción: para cada generador Σ aᵢpᵢ = 0,
        expresar el primer camino en términos de los demás.

        Regla: p_lead → −(1/a_lead) Σ_{i≠lead} aᵢ pᵢ
        """
        self._rules: list[tuple[Path, dict[Path, float]]] = []
        for gen in self.generators:
            if gen.is_zero():
                continue
            terms = list(gen.terms.items())
            lead_path, lead_coeff = terms[0]
            replacement = {}
            for p, c in terms[1:]:
                replacement[p] = -c / lead_coeff
            self._rules.append((lead_path, replacement))

    def reduce(self, element: PathAlgebraElement) -> PathAlgebraElement:
        """
        Reducir un elemento módulo I usando las reglas de reescritura.
        Retorna el representante canónico en kQ/I.
        """
        changed = True
        current = dict(element.terms)
        max_iter = 100
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for lead, replacement in self._rules:
                if lead in current and abs(current[lead]) > 1e-12:
                    coeff = current.pop(lead)
                    for p, c in replacement.items():
                        current[p] = current.get(p, 0.0) + coeff * c
                    changed = True
        return PathAlgebraElement(current)

    def is_admissible(self) -> bool:
        """
        Verificar si I es admisible: R_Q^m ⊆ I ⊆ R_Q^2 para algún m ≥ 2.
        (Definición 2.8)

        - I ⊆ R_Q^2: todos los generadores tienen grado ≥ 2.
        - R_Q^m ⊆ I: para quivers acíclicos, R_Q^m = 0 para m > diámetro,
          así que esta condición se cumple automáticamente.
        """
        # Verificar I ⊆ R_Q^2
        for gen in self.generators:
            for p in gen.terms:
                if p.length < 2:
                    return False
        return True


# ═══════════════════════════════════════════════════════════════════════
# QuotientAlgebra — Álgebra cociente kQ/I
# ═══════════════════════════════════════════════════════════════════════

class QuotientAlgebra:
    """
    Álgebra cociente kQ/I.

    Calcula dimensiones efectivas dim(e_i · (kQ/I)_k · e_j) que reemplazan
    (A^k)_{ij} en la tasa de impacto enriquecida cuando hay un ideal.
    (Proposición 3.3)
    """

    def __init__(self, algebra: PathAlgebra, ideal: Ideal):
        self.algebra = algebra
        self.ideal = ideal
        self._dim_cache: dict[tuple, int] = {}

    def dimension(self, source, target, length: int) -> int:
        """
        dim(e_{source} · (kQ/I)_{length} · e_{target}).

        Se calcula enumerando los caminos de longitud k de source a target,
        reduciéndolos módulo I, y contando las clases distintas no nulas.
        """
        key = (source, target, length)
        if key in self._dim_cache:
            return self._dim_cache[key]

        paths = self.algebra.paths_from_to(source, target, length)
        if not paths:
            self._dim_cache[key] = 0
            return 0

        # Reducir cada camino y contar representantes linealmente independientes
        reduced_vectors = []
        for p in paths:
            elem = PathAlgebraElement({p: 1.0})
            reduced = self.ideal.reduce(elem)
            vec = self._element_to_vector(reduced, paths)
            reduced_vectors.append(vec)

        if not reduced_vectors:
            self._dim_cache[key] = 0
            return 0

        # Rango de la matriz de vectores reducidos = dimensión del espacio cociente
        M = np.array(reduced_vectors)
        rank = int(np.linalg.matrix_rank(M, tol=1e-10))
        self._dim_cache[key] = rank
        return rank

    def _element_to_vector(self, element: PathAlgebraElement,
                           basis_paths: list[Path]) -> list[float]:
        """Convertir un elemento a coordenadas respecto a la base de caminos."""
        path_to_idx = {p: i for i, p in enumerate(basis_paths)}
        vec = [0.0] * len(basis_paths)
        for p, c in element.terms.items():
            if p in path_to_idx:
                vec[path_to_idx[p]] += c
        return vec

    def effective_walk_matrix(self, length: int) -> np.ndarray:
        """
        Matriz M_k donde M_k[i,j] = dim(e_i · (kQ/I)_k · e_j).
        Reemplaza A^k cuando se usa la tasa enriquecida con ideal.
        """
        n = self.algebra.quiver.n_vertices
        M = np.zeros((n, n), dtype=np.float64)
        verts = self.algebra.quiver.Q0
        for i, vi in enumerate(verts):
            for j, vj in enumerate(verts):
                M[i, j] = self.dimension(vi, vj, length)
        return M

    def weighted_impact_index(self, source, target, P, g_max: int) -> float:
        """
        I_P(c_i, c_j) = Σ_{g≥1} P(g) · dim(e_i · (kQ/I)_g · e_j).
        (Ecuación después de Definición 2.8)
        """
        total = 0.0
        for g in range(1, g_max + 1):
            total += P(g) * self.dimension(source, target, g)
        return total

    def total_dimension(self, max_length: Optional[int] = None) -> int:
        """dim(kQ/I) truncada a max_length."""
        Q = self.algebra.quiver
        if max_length is None:
            if Q.is_acyclic():
                max_length = Q.diameter()
            else:
                raise ValueError("Quiver tiene ciclos: especificar max_length")
        n = Q.n_vertices
        total = 0
        verts = Q.Q0
        for k in range(max_length + 1):
            for vi in verts:
                for vj in verts:
                    total += self.dimension(vi, vj, k)
        return total
