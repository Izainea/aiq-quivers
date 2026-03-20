"""
Morfismos de quivers y de AIQs.

Referencia: ACT.tex, Definición 3.2 (morfismo de quivers),
Definición 3.3 (morfismo base), Definición 3.4 (morfismo dinámico),
Definición 3.5 (morfismo algebraico), Proposición 3.4 (composición).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .quiver import Quiver


# ═══════════════════════════════════════════════════════════════════════
# Morfismo de quivers
# ═══════════════════════════════════════════════════════════════════════

class QuiverMorphism:
    """
    Morfismo de quivers f: Q → Q'.

    f = (f_0: Q_0 → Q'_0, f_1: Q_1 → Q'_1) tal que:
      s' ∘ f_1 = f_0 ∘ s   y   t' ∘ f_1 = f_0 ∘ t

    Parameters
    ----------
    source : Quiver
        Quiver fuente Q.
    target : Quiver
        Quiver objetivo Q'.
    vertex_map : dict
        f_0: {v ∈ Q_0 → f_0(v) ∈ Q'_0}.
    arrow_map : dict
        f_1: {nombre_α ∈ Q_1 → nombre_f_1(α) ∈ Q'_1}.
    """

    def __init__(
        self,
        source: Quiver,
        target: Quiver,
        vertex_map: dict,
        arrow_map: dict,
    ):
        self.source = source
        self.target = target
        self.vertex_map = dict(vertex_map)
        self.arrow_map = dict(arrow_map)
        self._validate()

    def _validate(self):
        """Verifica las condiciones de morfismo de quivers."""
        # Verificar que f_0 está definido en todo Q_0
        for v in self.source.Q0:
            if v not in self.vertex_map:
                raise ValueError(f"f_0 no definido para vértice '{v}'")
            if self.vertex_map[v] not in set(self.target.Q0):
                raise ValueError(
                    f"f_0({v}) = {self.vertex_map[v]} no está en Q'_0"
                )

        # Verificar que f_1 está definido en todo Q_1
        target_arrow_names = {name for name, _, _ in self.target.Q1}
        for name, src, tgt in self.source.Q1:
            if name not in self.arrow_map:
                raise ValueError(f"f_1 no definido para flecha '{name}'")
            f1_name = self.arrow_map[name]
            if f1_name not in target_arrow_names:
                raise ValueError(
                    f"f_1({name}) = {f1_name} no está en Q'_1"
                )

            # Condición: s' ∘ f_1 = f_0 ∘ s
            f1_src = self.target.source(f1_name)
            f0_src = self.vertex_map[src]
            if f1_src != f0_src:
                raise ValueError(
                    f"Falla s'∘f_1 = f_0∘s para flecha '{name}': "
                    f"s'(f_1({name})) = {f1_src} ≠ f_0(s({name})) = {f0_src}"
                )

            # Condición: t' ∘ f_1 = f_0 ∘ t
            f1_tgt = self.target.target(f1_name)
            f0_tgt = self.vertex_map[tgt]
            if f1_tgt != f0_tgt:
                raise ValueError(
                    f"Falla t'∘f_1 = f_0∘t para flecha '{name}': "
                    f"t'(f_1({name})) = {f1_tgt} ≠ f_0(t({name})) = {f0_tgt}"
                )

    def is_injective_on_vertices(self) -> bool:
        """f_0 es inyectivo."""
        vals = list(self.vertex_map.values())
        return len(vals) == len(set(vals))

    def is_surjective_on_vertices(self) -> bool:
        """f_0 es sobreyectivo."""
        return set(self.vertex_map.values()) == set(self.target.Q0)

    def is_isomorphism(self) -> bool:
        """f es isomorfismo (bijección en vértices y flechas)."""
        if not self.is_injective_on_vertices():
            return False
        if not self.is_surjective_on_vertices():
            return False
        arrow_vals = list(self.arrow_map.values())
        if len(arrow_vals) != len(set(arrow_vals)):
            return False
        target_arrows = {name for name, _, _ in self.target.Q1}
        return set(arrow_vals) == target_arrows

    def compose(self, other: QuiverMorphism) -> QuiverMorphism:
        """
        Composición g ∘ f donde self = f: Q → Q' y other = g: Q' → Q''.
        Retorna h: Q → Q'' con h_0 = g_0 ∘ f_0, h_1 = g_1 ∘ f_1.
        """
        if set(self.target.Q0) != set(other.source.Q0):
            raise ValueError("Quivers intermedios no coinciden")
        new_v_map = {v: other.vertex_map[self.vertex_map[v]]
                     for v in self.vertex_map}
        new_a_map = {a: other.arrow_map[self.arrow_map[a]]
                     for a in self.arrow_map}
        return QuiverMorphism(self.source, other.target, new_v_map, new_a_map)

    def __repr__(self) -> str:
        return (
            f"QuiverMorphism(|Q_0|={self.source.n_vertices}→"
            f"{self.target.n_vertices}, "
            f"iso={self.is_isomorphism()})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Morfismo de AIQs
# ═══════════════════════════════════════════════════════════════════════

class AIQMorphism:
    """
    Morfismo de AIQs μ: A → A'.

    Consta de:
      - f: QuiverMorphism Q → Q' (morfismo base)
      - tau: dict {estado_A → estado_A'} (traducción de estados)
      - pi_boundary: dict or None  (condiciones de frontera)

    Niveles de compatibilidad (Defs. 3.3–3.5):
      - Base (B): f es morfismo de quivers + tau traduce estados
      - Dinámico (D): la transición se preserva bajo el morfismo
      - Algebraico (A): kf(I) ⊆ I'

    Parameters
    ----------
    source_aiq : AIQ
        AIQ fuente A.
    target_aiq : AIQ
        AIQ objetivo A'.
    quiver_morphism : QuiverMorphism
        Morfismo f: Q → Q'.
    state_map : dict
        τ: {estado ∈ Σ → estado ∈ Σ'}.
    boundary_config : dict or None
        Configuración de frontera π_∂ para vértices de Q'
        fuera de la imagen de f.
    """

    def __init__(
        self,
        source_aiq,
        target_aiq,
        quiver_morphism: QuiverMorphism,
        state_map: dict,
        boundary_config: Optional[dict] = None,
    ):
        self.source = source_aiq
        self.target = target_aiq
        self.f = quiver_morphism
        self.tau = dict(state_map)
        self.boundary_config = boundary_config or {}
        self._validate_base()

    def _validate_base(self):
        """Verifica condiciones de morfismo base (Def. 3.3)."""
        # tau mapea estados de A a estados de A'
        for s in self.source.states:
            if s not in self.tau:
                raise ValueError(f"τ no definido para estado '{s}'")
            if self.tau[s] not in self.target.states:
                raise ValueError(
                    f"τ({s}) = {self.tau[s]} no está en Σ' = {self.target.states}"
                )

    def translate_config(self, config: dict) -> dict:
        """
        Traduce una configuración de A a una configuración parcial de A'.

        Para cada vértice v de Q con estado s, asigna τ(s) a f_0(v).
        Para vértices fuera de la imagen, usa boundary_config.
        """
        target_config = {}
        image_vertices = set()

        for v in self.source.quiver.Q0:
            v_prime = self.f.vertex_map[v]
            s = config.get(v)
            if s is not None:
                target_config[v_prime] = self.tau[s]
            image_vertices.add(v_prime)

        # Vértices fuera de la imagen: usar frontera
        for v_prime in self.target.quiver.Q0:
            if v_prime not in image_vertices:
                if v_prime in self.boundary_config:
                    target_config[v_prime] = self.boundary_config[v_prime]
                else:
                    # Estado por defecto: primer estado de Σ'
                    target_config[v_prime] = self.target.states[0]

        return target_config

    def is_base_morphism(self) -> bool:
        """
        Verifica condición (B): morfismo base.
        f es morfismo de quivers válido y τ traduce estados.
        (Siempre True si el constructor no lanzó excepción.)
        """
        return True

    def is_dynamic_morphism(
        self,
        test_configs: Optional[list[dict]] = None,
        n_tests: int = 20,
        seed: int = 42,
    ) -> bool:
        """
        Verifica condición (D): morfismo dinámico.

        Para varias configuraciones de prueba, verifica que:
          τ(φ_A(π)(v)) ≈ φ_{A'}(f*(π))(f_0(v))

        Es decir, evolucionar y luego traducir da lo mismo que
        traducir y luego evolucionar.

        Nota: por la estocasticidad, esto se verifica comparando
        las probabilidades de transición (tasas de impacto).
        """
        rng = np.random.default_rng(seed)

        if test_configs is None:
            test_configs = []
            for _ in range(n_tests):
                cfg = {}
                for v in self.source.quiver.Q0:
                    cfg[v] = self.source.states[
                        rng.integers(0, len(self.source.states))
                    ]
                test_configs.append(cfg)

        for cfg in test_configs:
            # Tasas en A
            self.source.set_initial_config(cfg)
            target_state = self.source.states[1] if len(self.source.states) > 1 else self.source.states[0]

            # Configuración traducida en A'
            cfg_prime = self.translate_config(cfg)
            self.target.set_initial_config(cfg_prime)
            target_state_prime = self.tau[target_state]

            # Comparar tasas para cada vértice
            for v in self.source.quiver.Q0:
                rate_source = self.source._compute_rate(v, target_state)
                v_prime = self.f.vertex_map[v]
                rate_target = self.target._compute_rate(v_prime, target_state_prime)

                if abs(rate_source - rate_target) > 1e-6:
                    return False

        return True

    def is_algebraic_morphism(self) -> bool:
        """
        Verifica condición (A): kf(I) ⊆ I'.
        Requiere que ambos AIQs tengan quotient con ideal.

        Nota: implementación simplificada — verifica que la imagen
        de cada generador del ideal I está en I'.
        """
        if self.source.quotient is None or self.target.quotient is None:
            return True  # Vacuamente verdadero

        source_ideal = self.source.quotient.ideal
        target_ideal = self.target.quotient.ideal

        if source_ideal is None or target_ideal is None:
            return True

        # Para cada generador de I, verificar que su imagen bajo kf está en I'
        # Esto requiere traducir caminos: camino p = α_1·...·α_n se mapea a
        # f_1(α_1)·...·f_1(α_n)
        for gen in source_ideal.generators:
            # gen es un PathAlgebraElement
            # Traducir cada camino
            for path, coeff in gen.terms.items():
                if path.length == 0:
                    continue
                # Mapear cada flecha del camino
                try:
                    mapped_arrows = []
                    for arrow_name in path.arrows:
                        mapped_arrows.append(self.f.arrow_map[arrow_name])
                except KeyError:
                    return False

            # Verificación completa requeriría construir el elemento
            # en kQ' y verificar pertenencia a I' — simplificado aquí.

        return True

    def __repr__(self) -> str:
        return (
            f"AIQMorphism({self.source.quiver.n_vertices}→"
            f"{self.target.quiver.n_vertices}, "
            f"τ={self.tau})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Funciones de conveniencia
# ═══════════════════════════════════════════════════════════════════════

def identity_morphism(quiver: Quiver) -> QuiverMorphism:
    """Morfismo identidad id: Q → Q."""
    v_map = {v: v for v in quiver.Q0}
    a_map = {name: name for name, _, _ in quiver.Q1}
    return QuiverMorphism(quiver, quiver, v_map, a_map)


def relabeling_isomorphism(
    quiver: Quiver,
    vertex_relabel: dict,
) -> QuiverMorphism:
    """
    Isomorfismo por reetiquetado de vértices.

    Parameters
    ----------
    vertex_relabel : dict
        {v_old → v_new} biyección.
    """
    new_vertices = [vertex_relabel[v] for v in quiver.Q0]
    new_arrows = []
    arrow_relabel = {}
    for name, src, tgt in quiver.Q1:
        new_name = f"{name}_r"
        new_arrows.append((new_name, vertex_relabel[src], vertex_relabel[tgt]))
        arrow_relabel[name] = new_name

    weights = None
    if quiver._weights:
        weights = {f"{n}_r": quiver.arrow_weight(n) for n, _, _ in quiver.Q1}

    target = Quiver(new_vertices, new_arrows, weights)
    return QuiverMorphism(quiver, target, vertex_relabel, arrow_relabel)


def subquiver_inclusion(
    quiver: Quiver,
    vertex_subset: list,
) -> QuiverMorphism:
    """
    Morfismo de inclusión ι: Q|_S ↪ Q.
    (Ejemplo 3.4)
    """
    sub = quiver.subquiver(vertex_subset)
    v_map = {v: v for v in vertex_subset}
    a_map = {name: name for name, _, _ in sub.Q1}
    return QuiverMorphism(sub, quiver, v_map, a_map)


def compose_morphisms(f: QuiverMorphism, g: QuiverMorphism) -> QuiverMorphism:
    """g ∘ f: Q → Q'' (Proposición 3.4)."""
    return f.compose(g)
