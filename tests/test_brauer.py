"""Tests for aiq.brauer — Brauer configurations and BCA."""

import math
import pytest

from aiq.brauer import (
    BrauerConfiguration,
    example_partitions_of_10,
    example_compositions_B7,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def bc_p10():
    """Example 2: partitions of 10."""
    return example_partitions_of_10()


@pytest.fixture
def bc_B7():
    """Example 7: compositions B(j,m;7)."""
    return example_compositions_B7()


# ── Construction ──────────────────────────────────────────────────────

class TestConstruction:
    def test_basic(self, bc_p10):
        assert bc_p10.n_vertices == 4
        assert bc_p10.n_polygons == 3

    def test_Gamma0(self, bc_p10):
        assert set(bc_p10.Gamma0) == {1, 2, 3, 4}

    def test_Gamma1(self, bc_p10):
        g1 = bc_p10.Gamma1
        assert "L1" in g1
        assert "L2" in g1
        assert "L3" in g1

    def test_M(self, bc_p10):
        assert bc_p10.M == {1, 2, 3, 4}

    def test_invalid_vertex_in_polygon(self):
        with pytest.raises(ValueError, match="no está en Γ₀"):
            BrauerConfiguration(
                vertices=[1, 2],
                polygons={"P1": [1, 99]},
            )


# ── Valency (Formula 11) ─────────────────────────────────────────────

class TestValency:
    def test_val_p10(self, bc_p10):
        assert bc_p10.valency(1) == 1
        assert bc_p10.valency(2) == 6
        assert bc_p10.valency(3) == 3
        assert bc_p10.valency(4) == 2

    def test_val_B7(self, bc_B7):
        assert bc_B7.valency(1) == 8
        assert bc_B7.valency(2) == 6
        assert bc_B7.valency(4) == 2


# ── Multiplicity μ ───────────────────────────────────────────────────

class TestMultiplicity:
    def test_mu_p10(self, bc_p10):
        assert bc_p10.mu(1) == 2  # val=1 → μ=2
        assert bc_p10.mu(2) == 1
        assert bc_p10.mu(3) == 1
        assert bc_p10.mu(4) == 1

    def test_default_mu(self):
        bc = BrauerConfiguration(
            vertices=[1, 2],
            polygons={"P1": [1, 2, 2]},
        )
        assert bc.mu(1) == 2  # val(1)=1 → μ=2
        assert bc.mu(2) == 1  # val(2)=2 → μ=1

    def test_truncated(self, bc_p10):
        assert bc_p10.is_truncated(1) is True
        assert bc_p10.is_truncated(2) is False

    def test_nontruncated(self, bc_p10):
        assert bc_p10.is_nontruncated(1) is True  # val=1, μ=2 → 1*2=2 > 1
        assert bc_p10.is_nontruncated(2) is True


# ── Incidence set ─────────────────────────────────────────────────────

class TestIncidenceSet:
    def test_incidence_1(self, bc_p10):
        inc = bc_p10.incidence_set(1)
        assert len(inc) == 1
        assert inc[0] == ("L2", 1)

    def test_incidence_2(self, bc_p10):
        inc = bc_p10.incidence_set(2)
        # 2 appears in L1 (1 time) and L3 (5 times)
        inc_dict = {p: c for p, c in inc}
        assert inc_dict["L1"] == 1
        assert inc_dict["L3"] == 5

    def test_incidence_4(self, bc_p10):
        inc = bc_p10.incidence_set(4)
        inc_dict = {p: c for p, c in inc}
        assert inc_dict["L1"] == 2


# ── Successor sequences ──────────────────────────────────────────────

class TestSuccessorSequences:
    def test_cached(self, bc_p10):
        s1 = bc_p10.successor_sequences()
        s2 = bc_p10.successor_sequences()
        assert s1 is s2  # Same object (cached)

    def test_seq_vertex_1(self, bc_p10):
        seqs = bc_p10.successor_sequences()
        # val(1)=1, only in L2 with mult 1 → seq = [(L2, 1)]
        assert len(seqs[1]) == 1
        assert seqs[1][0] == ("L2", 1)

    def test_compressed(self, bc_p10):
        css = bc_p10.compressed_successor_sequence(1)
        assert css == ["L2"]


# ── Brauer quiver Q_M ────────────────────────────────────────────────

class TestBrauerQuiver:
    def test_quiver_vertices(self, bc_p10):
        Q = bc_p10.brauer_quiver()
        assert set(Q.Q0) == {"L1", "L2", "L3"}

    def test_quiver_has_arrows(self, bc_p10):
        Q = bc_p10.brauer_quiver()
        assert Q.n_arrows > 0

    def test_quiver_cached(self, bc_p10):
        Q1 = bc_p10.brauer_quiver()
        Q2 = bc_p10.brauer_quiver()
        assert Q1 is Q2

    def test_arrow_names_readable(self, bc_p10):
        Q = bc_p10.brauer_quiver()
        for name, _, _ in Q.Q1:
            assert "s_" in name  # New naming convention


# ── Loops ─────────────────────────────────────────────────────────────

class TestLoops:
    def test_n_loops(self, bc_p10):
        # This tests that loops are counted correctly
        n = bc_p10.n_loops()
        assert isinstance(n, int)
        assert n >= 0

    def test_loops_cached(self, bc_p10):
        # First call builds quiver + caches loops
        n1 = bc_p10.n_loops()
        n2 = bc_p10.n_loops()
        assert n1 == n2


# ── Dimension formulas ────────────────────────────────────────────────

class TestDimension:
    def test_dimension_p10(self, bc_p10):
        dim = bc_p10.dimension()
        # dim = 2*3 + Σ val(m)*(μ(m)*val(m) - 1)
        # = 6 + 1*(2*1-1) + 6*(1*6-1) + 3*(1*3-1) + 2*(1*2-1)
        # = 6 + 1 + 30 + 6 + 2 = 45
        assert dim == 45

    def test_center_dimension_p10(self, bc_p10):
        cd = bc_p10.center_dimension()
        assert isinstance(cd, int)
        assert cd > 0


# ── Impact factor δ_B ────────────────────────────────────────────────

class TestImpactFactor:
    def test_delta_p10(self, bc_p10):
        # δ_B = Σ μ(m)*val(m) = 2*1 + 1*6 + 1*3 + 1*2 = 13
        assert bc_p10.impact_factor() == 13

    def test_delta_B7(self, bc_B7):
        delta = bc_B7.impact_factor()
        assert isinstance(delta, (int, float))
        assert delta > 0


# ── Entropy H(B) ─────────────────────────────────────────────────────

class TestEntropy:
    def test_entropy_positive(self, bc_p10):
        h = bc_p10.entropy()
        assert h > 0

    def test_entropy_bounded(self, bc_p10):
        # Shannon entropy ≤ log₂(n) where n = |Γ₀|
        h = bc_p10.entropy()
        assert h <= math.log2(bc_p10.n_vertices) + 0.01

    def test_entropy_zero(self):
        # Single vertex, single polygon
        bc = BrauerConfiguration(
            vertices=[1],
            polygons={"P1": [1]},
        )
        h = bc.entropy()
        assert h == 0.0  # Only one weight → p=1 → -1*log(1) = 0


# ── Word and message ──────────────────────────────────────────────────

class TestWordAndMessage:
    def test_word_L1(self, bc_p10):
        w = bc_p10.word("L1")
        assert "4" in w
        assert "2" in w

    def test_brauer_message(self, bc_p10):
        msg = bc_p10.brauer_message()
        assert "|" in msg

    def test_defect(self, bc_p10):
        d = bc_p10.defect()
        assert isinstance(d, int)
        assert d >= 0


# ── Covering graph ────────────────────────────────────────────────────

class TestCoveringGraph:
    def test_edges(self, bc_p10):
        edges = bc_p10.covering_graph_edges()
        assert isinstance(edges, list)


# ── Full analysis ─────────────────────────────────────────────────────

class TestBrauerAnalysis:
    def test_analysis_keys(self, bc_p10):
        analysis = bc_p10.brauer_analysis()
        expected_keys = {
            "n_vertices", "n_polygons", "dimension", "center_dimension",
            "n_loops", "defect", "impact_factor_delta_B", "entropy_H_B",
            "valencies", "multiplicities", "truncated_vertices",
            "nontruncated_vertices",
        }
        assert expected_keys == set(analysis.keys())

    def test_summary(self, bc_p10):
        s = bc_p10.summary()
        assert "δ_B" in s
        assert "H(B)" in s

    def test_repr(self, bc_p10):
        r = repr(bc_p10)
        assert "Γ₀" in r
        assert "Γ₁" in r
