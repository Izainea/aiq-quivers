"""Tests for aiq.morphisms — Quiver and AIQ morphisms."""

import pytest

from aiq.quiver import Quiver
from aiq.morphisms import (
    QuiverMorphism,
    AIQMorphism,
    identity_morphism,
    relabeling_isomorphism,
    subquiver_inclusion,
    compose_morphisms,
)
from aiq.automaton import AIQ


@pytest.fixture
def q4():
    return Quiver(
        [1, 2, 3, 4],
        [("α₁", 1, 2), ("α₂", 1, 2), ("γ", 1, 3), ("β", 2, 4), ("δ", 3, 4)],
    )


@pytest.fixture
def q_small():
    return Quiver([1, 2], [("a", 1, 2)])


# ── QuiverMorphism ────────────────────────────────────────────────────

class TestQuiverMorphism:
    def test_identity(self, q4):
        f = identity_morphism(q4)
        assert f.is_isomorphism()

    def test_invalid_vertex_map(self, q_small):
        q2 = Quiver([1, 2], [("b", 1, 2)])
        with pytest.raises(ValueError):
            QuiverMorphism(q_small, q2, vertex_map={1: 1}, arrow_map={"a": "b"})

    def test_invalid_arrow_compatibility(self):
        q1 = Quiver([1, 2], [("a", 1, 2)])
        q2 = Quiver([1, 2], [("b", 2, 1)])
        with pytest.raises(ValueError, match="Falla"):
            QuiverMorphism(q1, q2, vertex_map={1: 1, 2: 2}, arrow_map={"a": "b"})

    def test_injective(self, q4):
        f = identity_morphism(q4)
        assert f.is_injective_on_vertices()

    def test_surjective(self, q4):
        f = identity_morphism(q4)
        assert f.is_surjective_on_vertices()


# ── Relabeling ────────────────────────────────────────────────────────

class TestRelabeling:
    def test_relabeling(self, q_small):
        relabel = {1: "A", 2: "B"}
        f = relabeling_isomorphism(q_small, relabel)
        assert f.is_isomorphism()
        assert f.vertex_map == relabel

    def test_relabeled_quiver(self, q_small):
        relabel = {1: "A", 2: "B"}
        f = relabeling_isomorphism(q_small, relabel)
        assert set(f.target.Q0) == {"A", "B"}


# ── Subquiver inclusion ──────────────────────────────────────────────

class TestSubquiverInclusion:
    def test_inclusion(self, q4):
        iota = subquiver_inclusion(q4, [1, 2])
        assert iota.source.n_vertices == 2
        assert iota.target is q4
        assert iota.is_injective_on_vertices()

    def test_inclusion_preserves_arrows(self, q4):
        iota = subquiver_inclusion(q4, [1, 2])
        assert iota.source.n_arrows == 2  # α₁, α₂


# ── Composition ───────────────────────────────────────────────────────

class TestComposition:
    def test_compose_identities(self, q4):
        f = identity_morphism(q4)
        g = identity_morphism(q4)
        h = compose_morphisms(f, g)
        assert h.is_isomorphism()

    def test_compose_method(self, q4):
        f = identity_morphism(q4)
        g = identity_morphism(q4)
        h = f.compose(g)
        assert h.is_isomorphism()


# ── AIQMorphism ───────────────────────────────────────────────────────

class TestAIQMorphism:
    def test_base_morphism(self, q4):
        aiq1 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq2 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        f = identity_morphism(q4)
        tau = {"S": "S", "I": "I"}
        mu = AIQMorphism(aiq1, aiq2, f, tau)
        assert mu.is_base_morphism()

    def test_translate_config(self, q4):
        aiq1 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq2 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        f = identity_morphism(q4)
        tau = {"S": "S", "I": "I"}
        mu = AIQMorphism(aiq1, aiq2, f, tau)
        cfg = {1: "I", 2: "S", 3: "S", 4: "S"}
        translated = mu.translate_config(cfg)
        assert translated == cfg

    def test_invalid_state_map(self, q4):
        aiq1 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq2 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        f = identity_morphism(q4)
        with pytest.raises(ValueError, match="τ no definido"):
            AIQMorphism(aiq1, aiq2, f, {"S": "S"})

    def test_repr(self, q4):
        aiq1 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq2 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        f = identity_morphism(q4)
        tau = {"S": "S", "I": "I"}
        mu = AIQMorphism(aiq1, aiq2, f, tau)
        r = repr(mu)
        assert "4→4" in r


# ── is_algebraic_morphism (condición A: kf(I) ⊆ I') ──────────────────────

class TestAlgebraicMorphism:
    def _aiq_with_ideal(self, q, generators):
        from aiq.path_algebra import PathAlgebra, Ideal, QuotientAlgebra
        kQ = PathAlgebra(q)
        quotient = QuotientAlgebra(kQ, Ideal(kQ, generators))
        return AIQ(q, states=["S", "I"], evolution_rule="SI",
                   g_max=2, quotient=quotient)

    def test_vacuously_true_without_quotient(self, q4):
        aiq1 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq2 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        mu = AIQMorphism(aiq1, aiq2, identity_morphism(q4), {"S": "S", "I": "I"})
        assert mu.is_algebraic_morphism() is True

    def test_identity_maps_ideal_into_itself(self, q4):
        # I = I' = ⟨α₁ − α₂⟩; bajo la identidad la imagen del generador
        # se reduce a 0 en I', así que kf(I) ⊆ I'.
        from aiq.path_algebra import Path, PathAlgebraElement
        gen = PathAlgebraElement({
            Path(q4, arrows=["α₁"]): 1.0,
            Path(q4, arrows=["α₂"]): -1.0,
        })
        aiq1 = self._aiq_with_ideal(q4, [gen])
        aiq2 = self._aiq_with_ideal(q4, [gen])
        mu = AIQMorphism(aiq1, aiq2, identity_morphism(q4), {"S": "S", "I": "I"})
        assert mu.is_algebraic_morphism() is True

    def test_image_not_in_target_ideal(self, q4):
        # I = ⟨α₁ − α₂⟩ pero I' = ⟨0⟩ (vacío): la imagen α₁ − α₂ no es 0
        # en kQ'/I', así que la condición (A) falla.
        from aiq.path_algebra import Path, PathAlgebraElement
        gen = PathAlgebraElement({
            Path(q4, arrows=["α₁"]): 1.0,
            Path(q4, arrows=["α₂"]): -1.0,
        })
        aiq1 = self._aiq_with_ideal(q4, [gen])
        aiq2 = self._aiq_with_ideal(q4, [])  # ideal trivial
        mu = AIQMorphism(aiq1, aiq2, identity_morphism(q4), {"S": "S", "I": "I"})
        assert mu.is_algebraic_morphism() is False
