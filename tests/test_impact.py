"""Tests for aiq.impact — Impact metrics, SFV, and rates."""

import numpy as np
import pytest

from aiq.quiver import Quiver
from aiq.impact import (
    impact_degree,
    impact_vector,
    impact_vector_matrix,
    FundamentalNeighborhoodSystem,
    _LayerCache,
    impact_rate_simple,
    impact_rate_enriched,
    impact_rate_signed,
    effective_transition_probability,
)


@pytest.fixture
def q4():
    return Quiver(
        [1, 2, 3, 4],
        [("α₁", 1, 2), ("α₂", 1, 2), ("γ", 1, 3), ("β", 2, 4), ("δ", 3, 4)],
    )


@pytest.fixture
def config_SI():
    return {1: "I", 2: "S", 3: "S", 4: "S"}


# ── Impact degree ─────────────────────────────────────────────────────

class TestImpactDegree:
    def test_adjacent(self, q4):
        assert impact_degree(q4, 1, 2) == 1

    def test_two_hops(self, q4):
        assert impact_degree(q4, 1, 4) == 2

    def test_unreachable(self, q4):
        assert impact_degree(q4, 4, 1) == float("inf")

    def test_self(self, q4):
        assert impact_degree(q4, 1, 1) == 0


# ── Impact vector ─────────────────────────────────────────────────────

class TestImpactVector:
    def test_shape(self, q4):
        vec = impact_vector(q4, 1, 4)
        assert len(vec) == 3  # diameter=2, so max_k=2

    def test_values(self, q4):
        vec = impact_vector(q4, 1, 4, max_k=2)
        assert vec[0] == 0  # no length-0 walk from 1 to 4
        assert vec[1] == 0  # no length-1 walk from 1 to 4
        assert vec[2] == 3  # α₁β, α₂β, γδ

    def test_matrix(self, q4):
        V = impact_vector_matrix(q4)
        assert V.shape == (4, 4, 3)
        assert V[0, 3, 2] == 3


# ── Fundamental Neighborhood System ──────────────────────────────────

class TestFNS:
    def test_layer_0(self, q4):
        fns = FundamentalNeighborhoodSystem(q4, 4, g_max=2, direction="in")
        assert fns.layer(0) == {4}

    def test_layer_1(self, q4):
        fns = FundamentalNeighborhoodSystem(q4, 4, g_max=2, direction="in")
        layer1 = fns.layer(1)
        assert 2 in layer1
        assert 3 in layer1

    def test_layer_2(self, q4):
        fns = FundamentalNeighborhoodSystem(q4, 4, g_max=2, direction="in")
        assert fns.layer(2) == {1}

    def test_ball(self, q4):
        fns = FundamentalNeighborhoodSystem(q4, 4, g_max=2, direction="in")
        assert fns.A(0) == {4}
        assert fns.A(1) == {2, 3, 4}
        assert fns.A(2) == {1, 2, 3, 4}

    def test_Delta(self, q4):
        fns = FundamentalNeighborhoodSystem(q4, 4, g_max=2, direction="in")
        assert fns.Delta(0) == 1
        assert fns.Delta(1) == 2
        assert fns.Delta(2) == 1

    def test_topologically_trapped(self, q4):
        fns_source = FundamentalNeighborhoodSystem(q4, 1, g_max=2, direction="in")
        assert fns_source.is_topologically_trapped() is True

        fns_sink = FundamentalNeighborhoodSystem(q4, 4, g_max=2, direction="in")
        assert fns_sink.is_topologically_trapped() is False

    def test_precomputed_dist_matrix(self, q4):
        D = q4.distance_matrix()
        fns = FundamentalNeighborhoodSystem(q4, 4, g_max=2, direction="in", _dist_matrix=D)
        assert fns.layer(1) == {2, 3}


# ── LayerCache ────────────────────────────────────────────────────────

class TestLayerCache:
    def test_basic(self, q4):
        cache = _LayerCache(q4, g_max=2, direction="in")
        # Cell 4 (idx 3), layer 1 should have vertices 2 and 3
        layer = cache.layer(3, 1)
        labels = {v for v, _ in layer}
        assert labels == {2, 3}

    def test_layer_0(self, q4):
        cache = _LayerCache(q4, g_max=2, direction="in")
        layer = cache.layer(3, 0)
        labels = {v for v, _ in layer}
        assert labels == {4}

    def test_out_of_range(self, q4):
        cache = _LayerCache(q4, g_max=2, direction="in")
        assert cache.layer(3, 5) == []


# ── Impact rates ──────────────────────────────────────────────────────

class TestImpactRates:
    def test_simple_rate_no_infected_neighbors(self, q4, config_SI):
        # Vertex 1 is I, but it's a source — no one influences it
        rate = impact_rate_simple(
            q4, 1, config_SI, "I", beta=1.0, alpha=1.0,
            P=lambda g: 1.0 / g, g_max=2,
        )
        assert rate == 0.0  # source has no incoming influence

    def test_simple_rate_influenced(self, q4, config_SI):
        # Vertex 4 has layer 1={2,3} (both S) and layer 2={1} (I)
        rate = impact_rate_simple(
            q4, 4, config_SI, "I", beta=1.0, alpha=1.0,
            P=lambda g: 1.0 / g, g_max=2,
        )
        # Layer 1: σ=0 (no I in {2,3}), layer 2: σ=1 (1 is I), Δ=1
        # rate = (0/2)*1 + (1/1)*0.5 = 0.5
        assert abs(rate - 0.5) < 1e-10

    def test_simple_rate_with_cache(self, q4, config_SI):
        cache = _LayerCache(q4, g_max=2, direction="in")
        rate = impact_rate_simple(
            q4, 4, config_SI, "I", beta=1.0, alpha=1.0,
            P=lambda g: 1.0 / g, g_max=2, _layer_cache=cache,
        )
        assert abs(rate - 0.5) < 1e-10

    def test_enriched_rate(self, q4, config_SI):
        rate = impact_rate_enriched(
            q4, 4, config_SI, "I", beta=1.0, alpha=1.0,
            P=lambda g: 1.0 / g, g_max=2,
        )
        assert rate >= 0

    def test_signed_rate(self):
        q = Quiver(
            [1, 2, 3],
            [("a", 1, 2), ("b", 1, 3)],
            weights={"a": 1, "b": -1},
        )
        config = {1: "I", 2: "S", 3: "S"}
        rate = impact_rate_signed(
            q, 2, config, "I", beta=1.0, alpha=1.0,
            P=lambda g: 1.0, g_max=1,
        )
        assert isinstance(rate, float)


# ── Effective transition probability ──────────────────────────────────

class TestEffectiveProb:
    def test_clamp_normal(self):
        assert effective_transition_probability(0.5) == 0.5

    def test_clamp_negative(self):
        assert effective_transition_probability(-0.3) == 0.0

    def test_clamp_above_one(self):
        assert effective_transition_probability(1.5) == 1.0

    def test_clamp_zero(self):
        assert effective_transition_probability(0.0) == 0.0

    def test_clamp_one(self):
        assert effective_transition_probability(1.0) == 1.0
