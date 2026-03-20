"""Tests for aiq.quiver — Quiver class."""

import numpy as np
import pytest

from aiq.quiver import Quiver


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def q4():
    """4-node quiver from Example 2.3: two arrows 1→2, plus 1→3, 2→4, 3→4."""
    return Quiver(
        [1, 2, 3, 4],
        [("α₁", 1, 2), ("α₂", 1, 2), ("γ", 1, 3), ("β", 2, 4), ("δ", 3, 4)],
    )


@pytest.fixture
def q_cycle():
    """Simple 3-cycle: 1→2→3→1."""
    return Quiver([1, 2, 3], [("a", 1, 2), ("b", 2, 3), ("c", 3, 1)])


@pytest.fixture
def q_single():
    """Single vertex, no arrows."""
    return Quiver(["x"], [])


# ── Construction ──────────────────────────────────────────────────────

class TestConstruction:
    def test_basic(self, q4):
        assert q4.n_vertices == 4
        assert q4.n_arrows == 5

    def test_invalid_source(self):
        with pytest.raises(ValueError, match="Fuente"):
            Quiver([1, 2], [("a", 99, 2)])

    def test_invalid_target(self):
        with pytest.raises(ValueError, match="Objetivo"):
            Quiver([1, 2], [("a", 1, 99)])

    def test_empty_quiver(self):
        q = Quiver([], [])
        assert q.n_vertices == 0
        assert q.n_arrows == 0

    def test_single_vertex(self, q_single):
        assert q_single.n_vertices == 1
        assert q_single.n_arrows == 0


# ── Properties ────────────────────────────────────────────────────────

class TestProperties:
    def test_Q0(self, q4):
        assert q4.Q0 == [1, 2, 3, 4]

    def test_Q1_length(self, q4):
        assert len(q4.Q1) == 5

    def test_vertex_index(self, q4):
        assert q4.vertex_index(1) == 0
        assert q4.vertex_index(4) == 3

    def test_source_target(self, q4):
        assert q4.source("α₁") == 1
        assert q4.target("α₁") == 2
        assert q4.source("δ") == 3
        assert q4.target("δ") == 4

    def test_source_not_found(self, q4):
        with pytest.raises(KeyError):
            q4.source("nonexistent")

    def test_arrow_weight_default(self, q4):
        assert q4.arrow_weight("α₁") == 1

    def test_arrow_weight_signed(self):
        q = Quiver([1, 2], [("a", 1, 2)], weights={"a": -1})
        assert q.arrow_weight("a") == -1


# ── Degrees (O(1) with pre-computation) ──────────────────────────────

class TestDegrees:
    def test_in_degree(self, q4):
        assert q4.in_degree(1) == 0
        assert q4.in_degree(2) == 2
        assert q4.in_degree(3) == 1
        assert q4.in_degree(4) == 2

    def test_out_degree(self, q4):
        assert q4.out_degree(1) == 3
        assert q4.out_degree(2) == 1
        assert q4.out_degree(3) == 1
        assert q4.out_degree(4) == 0

    def test_is_source(self, q4):
        assert q4.is_source(1) is True
        assert q4.is_source(2) is False

    def test_is_sink(self, q4):
        assert q4.is_sink(4) is True
        assert q4.is_sink(1) is False

    def test_predecessors(self, q4):
        assert sorted(q4.predecessors_direct(4)) == [2, 3]
        assert q4.predecessors_direct(1) == []

    def test_successors(self, q4):
        # Vertex 1 has 3 outgoing arrows: two to 2 and one to 3
        succ = q4.successors_direct(1)
        assert succ.count(2) == 2
        assert succ.count(3) == 1
        assert q4.successors_direct(4) == []


# ── Adjacency matrix ─────────────────────────────────────────────────

class TestAdjacencyMatrix:
    def test_shape(self, q4):
        A = q4.adjacency_matrix()
        assert A.shape == (4, 4)

    def test_values(self, q4):
        A = q4.adjacency_matrix()
        assert A[0, 1] == 2  # two arrows 1→2
        assert A[0, 2] == 1  # one arrow 1→3
        assert A[1, 3] == 1  # 2→4
        assert A[2, 3] == 1  # 3→4
        assert A[3, 0] == 0  # no arrow 4→1

    def test_power(self, q4):
        A2 = q4.adjacency_power(2)
        # (A^2)_{0,3} = walks of length 2 from 1 to 4 = α₁·β + α₂·β + γ·δ = 3
        assert A2[0, 3] == 3

    def test_power_zero(self, q4):
        A0 = q4.adjacency_power(0)
        np.testing.assert_array_equal(A0, np.eye(4))

    def test_signed_matrix(self):
        q = Quiver([1, 2, 3], [("a", 1, 2), ("b", 1, 3)], weights={"a": 1, "b": -1})
        W = q.signed_adjacency_matrix()
        assert W[0, 1] == 1
        assert W[0, 2] == -1

    def test_returns_copy(self, q4):
        A1 = q4.adjacency_matrix()
        A2 = q4.adjacency_matrix()
        A1[0, 0] = 999
        assert A2[0, 0] != 999


# ── Distances and diameter ────────────────────────────────────────────

class TestDistances:
    def test_shortest_path(self, q4):
        assert q4.shortest_path_length(1, 2) == 1
        assert q4.shortest_path_length(1, 4) == 2
        assert q4.shortest_path_length(4, 1) == float("inf")

    def test_diameter(self, q4):
        assert q4.diameter() == 2

    def test_cycle_diameter(self, q_cycle):
        assert q_cycle.diameter() == 2

    def test_single_vertex_diameter(self, q_single):
        assert q_single.diameter() == 0

    def test_distance_matrix_shape(self, q4):
        D = q4.distance_matrix()
        assert D.shape == (4, 4)
        assert D[0, 0] == 0


# ── Topological properties ────────────────────────────────────────────

class TestTopology:
    def test_is_acyclic_dag(self, q4):
        assert q4.is_acyclic() is True

    def test_is_acyclic_cycle(self, q_cycle):
        assert q_cycle.is_acyclic() is False

    def test_acyclic_cached(self, q4):
        # Should be cached after first call
        assert q4.is_acyclic() is True
        assert q4.is_acyclic() is True


# ── Operations ────────────────────────────────────────────────────────

class TestOperations:
    def test_opposite(self, q4):
        Q_op = q4.opposite()
        assert Q_op.n_vertices == 4
        assert Q_op.n_arrows == 5
        # 4 is now a source in Q^op
        assert Q_op.is_source(4) is True
        assert Q_op.is_sink(1) is True

    def test_symmetrize(self, q4):
        Q_sym = q4.symmetrize()
        assert Q_sym.n_vertices == 4
        assert Q_sym.n_arrows >= q4.n_arrows

    def test_subquiver(self, q4):
        sub = q4.subquiver([1, 2])
        assert sub.n_vertices == 2
        assert sub.n_arrows == 2  # α₁ and α₂

    def test_influence_boundary(self, q4):
        boundary = q4.influence_boundary([4], g_max=1)
        assert 2 in boundary
        assert 3 in boundary

    def test_from_adjacency_matrix(self):
        A = np.array([[0, 2, 1], [0, 0, 1], [0, 0, 0]])
        q = Quiver.from_adjacency_matrix(A, labels=["a", "b", "c"])
        assert q.n_vertices == 3
        assert q.n_arrows == 4
        np.testing.assert_array_equal(q.adjacency_matrix(), A)


# ── NetworkX integration ──────────────────────────────────────────────

class TestNetworkX:
    def test_roundtrip(self, q4):
        G = q4.to_networkx()
        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 5

    def test_from_networkx(self, q4):
        G = q4.to_networkx()
        q2 = Quiver.from_networkx(G)
        assert q2.n_vertices == q4.n_vertices
        assert q2.n_arrows == q4.n_arrows


# ── Repr ──────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr(self, q4):
        r = repr(q4)
        assert "Q_0|=4" in r
        assert "Q_1|=5" in r

    def test_summary(self, q4):
        s = q4.summary()
        assert "DAG: True" in s
        assert "Diámetro: 2" in s
