"""Tests for aiq.automaton — AIQ automaton class."""

import pytest
import numpy as np

from aiq.quiver import Quiver
from aiq.automaton import AIQ


@pytest.fixture
def q4():
    return Quiver(
        [1, 2, 3, 4],
        [("α₁", 1, 2), ("α₂", 1, 2), ("γ", 1, 3), ("β", 2, 4), ("δ", 3, 4)],
    )


@pytest.fixture
def config_SI():
    return {1: "I", 2: "S", 3: "S", 4: "S"}


# ── Construction ──────────────────────────────────────────────────────

class TestConstruction:
    def test_basic(self, q4):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI")
        assert aiq.quiver is q4
        assert aiq.states == ["S", "I"]
        assert aiq.evolution_rule == "SI"

    def test_invalid_rule(self, q4):
        with pytest.raises(ValueError, match="desconocida"):
            AIQ(q4, states=["S", "I"], evolution_rule="INVALID")

    def test_default_P(self, q4):
        aiq = AIQ(q4, states=["S", "I"])
        assert aiq.P(0) == 1.0
        assert aiq.P(1) == 0.5

    def test_modes(self, q4):
        for mode in ["simple", "enriched", "signed"]:
            aiq = AIQ(q4, states=["S", "I"], impact_mode=mode)
            assert aiq.impact_mode == mode


# ── Configuration ─────────────────────────────────────────────────────

class TestConfig:
    def test_set_initial_config(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"])
        aiq.set_initial_config(config_SI)
        assert aiq.config == config_SI
        assert aiq.time == 0

    def test_missing_vertex(self, q4):
        aiq = AIQ(q4, states=["S", "I"])
        with pytest.raises(ValueError, match="Falta"):
            aiq.set_initial_config({1: "S", 2: "S"})

    def test_invalid_state(self, q4):
        aiq = AIQ(q4, states=["S", "I"])
        with pytest.raises(ValueError, match="no está en"):
            aiq.set_initial_config({1: "X", 2: "S", 3: "S", 4: "S"})

    def test_fixed_states(self, q4):
        aiq = AIQ(q4, states=["S", "I"], fixed_states={1: "I"})
        config = {1: "S", 2: "S", 3: "S", 4: "S"}
        aiq.set_initial_config(config)
        assert aiq.config[1] == "I"


# ── SI rule ───────────────────────────────────────────────────────────

class TestSI:
    def test_si_propagation(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", beta=10.0, g_max=2)
        aiq.set_initial_config(config_SI)
        aiq.run(10, seed=42)
        # With high beta, all should eventually become I
        counts = aiq.state_counts()
        assert counts["I"] == 4

    def test_si_source_stays(self, q4, config_SI):
        """Vertex 1 is a source and starts as I — should stay I."""
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", beta=1.0, g_max=2)
        aiq.set_initial_config(config_SI)
        aiq.run(5, seed=42)
        for cfg in aiq.orbit:
            assert cfg[1] == "I"

    def test_si_deterministic_seed(self, q4, config_SI):
        """Same seed produces same orbit."""
        aiq1 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq1.set_initial_config(config_SI)
        aiq1.run(5, seed=123)

        aiq2 = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq2.set_initial_config(config_SI)
        aiq2.run(5, seed=123)

        assert aiq1.orbit == aiq2.orbit


# ── SIS rule ──────────────────────────────────────────────────────────

class TestSIS:
    def test_sis_recovery(self, q4):
        config = {1: "I", 2: "I", 3: "I", 4: "I"}
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SIS",
                  beta=0.0, recovery_prob=1.0, g_max=2)
        aiq.set_initial_config(config)
        aiq.step(seed=42)
        # Vertex 1 is source — recovery should work
        # With recovery_prob=1.0 and beta=0, all I→S except possibly re-infected
        counts = aiq.state_counts()
        assert counts["S"] > 0


# ── SIR rule ──────────────────────────────────────────────────────────

class TestSIR:
    def test_sir_recovery(self, q4):
        config = {1: "I", 2: "I", 3: "I", 4: "I"}
        aiq = AIQ(q4, states=["S", "I", "R"], evolution_rule="SIR",
                  beta=0.0, recovery_prob=1.0, g_max=2)
        aiq.set_initial_config(config)
        aiq.step(seed=42)
        counts = aiq.state_counts()
        assert counts["R"] == 4  # All I→R with prob=1

    def test_sir_absorbing(self, q4):
        """R state is absorbing."""
        config = {1: "R", 2: "R", 3: "R", 4: "R"}
        aiq = AIQ(q4, states=["S", "I", "R"], evolution_rule="SIR", g_max=2)
        aiq.set_initial_config(config)
        aiq.step(seed=42)
        assert aiq.state_counts() == {"S": 0, "I": 0, "R": 4}


# ── Orbit and statistics ──────────────────────────────────────────────

class TestOrbitAndStats:
    def test_orbit_length(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq.set_initial_config(config_SI)
        aiq.run(5, seed=42)
        assert len(aiq.orbit) == 6  # initial + 5 steps
        assert aiq.time == 5

    def test_state_counts(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq.set_initial_config(config_SI)
        counts = aiq.state_counts()
        assert counts["I"] == 1
        assert counts["S"] == 3

    def test_orbit_table(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq.set_initial_config(config_SI)
        aiq.run(3, seed=42)
        df = aiq.orbit_table()
        assert df.shape[0] == 4  # 4 time steps (0,1,2,3)
        assert df.shape[1] == 4  # 4 vertices

    def test_orbit_counts_table(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq.set_initial_config(config_SI)
        aiq.run(3, seed=42)
        df = aiq.orbit_counts_table()
        assert "S" in df.columns
        assert "I" in df.columns

    def test_run_statistics(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        stats = aiq.run_statistics(config_SI, n_steps=3, n_runs=5, seed=42)
        assert "mean" in stats.columns
        assert "std" in stats.columns
        assert len(stats) == 4 * 2  # 4 time steps × 2 states


# ── Topological utilities ─────────────────────────────────────────────

class TestTopological:
    def test_trapped_vertices(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq.set_initial_config(config_SI)
        trapped = aiq.topologically_trapped_vertices()
        assert trapped == [1]

    def test_impact_rates_snapshot(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq.set_initial_config(config_SI)
        rates = aiq.impact_rates_snapshot()
        assert 1 in rates
        assert 4 in rates
        assert all(isinstance(v, float) for v in rates.values())


# ── Repr ──────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr(self, q4, config_SI):
        aiq = AIQ(q4, states=["S", "I"], evolution_rule="SI", g_max=2)
        aiq.set_initial_config(config_SI)
        r = repr(aiq)
        assert "SI" in r
        assert "Q_0|=4" in r
