"""Tests for aiq.path_algebra — Path, PathAlgebra, Ideal, QuotientAlgebra."""

import pytest
import numpy as np

from aiq.quiver import Quiver
from aiq.path_algebra import Path, PathAlgebraElement, PathAlgebra, Ideal, QuotientAlgebra


@pytest.fixture
def q4():
    return Quiver(
        [1, 2, 3, 4],
        [("α₁", 1, 2), ("α₂", 1, 2), ("γ", 1, 3), ("β", 2, 4), ("δ", 3, 4)],
    )


@pytest.fixture
def pa(q4):
    return PathAlgebra(q4)


# ── Path ──────────────────────────────────────────────────────────────

class TestPath:
    def test_trivial_path(self, q4):
        p = Path(q4, vertex=1)
        assert p.length == 0
        assert p.source == 1
        assert p.target == 1

    def test_single_arrow(self, q4):
        p = Path(q4, arrows=["α₁"])
        assert p.length == 1
        assert p.source == 1
        assert p.target == 2

    def test_concatenation(self, q4):
        p1 = Path(q4, arrows=["α₁"])
        p2 = Path(q4, arrows=["β"])
        p3 = p1.concatenate(p2)
        assert p3 is not None
        assert p3.length == 2
        assert p3.source == 1
        assert p3.target == 4

    def test_concatenation_incompatible(self, q4):
        p1 = Path(q4, arrows=["β"])   # 2→4
        p2 = Path(q4, arrows=["γ"])   # 1→3
        result = p1.concatenate(p2)
        assert result is None

    def test_trivial_concatenation(self, q4):
        e1 = Path(q4, vertex=1)
        p = Path(q4, arrows=["α₁"])
        assert e1.concatenate(p) == p
        e2 = Path(q4, vertex=2)
        assert p.concatenate(e2) == p

    def test_equality(self, q4):
        p1 = Path(q4, arrows=["α₁", "β"])
        p2 = Path(q4, arrows=["α₁", "β"])
        assert p1 == p2

    def test_hash(self, q4):
        p1 = Path(q4, arrows=["α₁", "β"])
        p2 = Path(q4, arrows=["α₂", "β"])
        assert hash(p1) != hash(p2)

    def test_sign_default(self, q4):
        p = Path(q4, arrows=["α₁", "β"])
        assert p.sign() == 1

    def test_sign_weighted(self):
        q = Quiver([1, 2, 3], [("a", 1, 2), ("b", 2, 3)], weights={"a": 1, "b": -1})
        p = Path(q, arrows=["a", "b"])
        assert p.sign() == -1

    def test_invalid_construction(self, q4):
        with pytest.raises(ValueError):
            Path(q4, arrows=["α₁"], vertex=1)
        with pytest.raises(ValueError):
            Path(q4)
        with pytest.raises(ValueError):
            Path(q4, arrows=[])

    def test_repr(self, q4):
        assert "e_1" in repr(Path(q4, vertex=1))
        assert "α₁" in repr(Path(q4, arrows=["α₁"]))


# ── PathAlgebraElement ────────────────────────────────────────────────

class TestPathAlgebraElement:
    def test_zero(self):
        e = PathAlgebraElement({})
        assert e.is_zero()

    def test_add(self, q4):
        p1 = Path(q4, arrows=["α₁"])
        p2 = Path(q4, arrows=["α₂"])
        e1 = PathAlgebraElement({p1: 1.0})
        e2 = PathAlgebraElement({p2: 1.0})
        e3 = e1 + e2
        assert len(e3.terms) == 2

    def test_sub(self, q4):
        p = Path(q4, arrows=["α₁"])
        e = PathAlgebraElement({p: 1.0})
        result = e - e
        assert result.is_zero()

    def test_mul(self, q4):
        p1 = Path(q4, arrows=["α₁"])
        p2 = Path(q4, arrows=["β"])
        e1 = PathAlgebraElement({p1: 1.0})
        e2 = PathAlgebraElement({p2: 1.0})
        prod = e1 * e2
        assert not prod.is_zero()
        path_result = list(prod.terms.keys())[0]
        assert path_result.length == 2

    def test_scalar_mul(self, q4):
        p = Path(q4, arrows=["α₁"])
        e = PathAlgebraElement({p: 1.0})
        e2 = 3.0 * e
        assert list(e2.terms.values())[0] == 3.0

    def test_homogeneous(self, q4):
        p = Path(q4, arrows=["α₁"])
        e = PathAlgebraElement({p: 1.0})
        assert e.is_homogeneous() == 1

    def test_not_homogeneous(self, q4):
        p1 = Path(q4, arrows=["α₁"])
        p2 = Path(q4, vertex=1)
        e = PathAlgebraElement({p1: 1.0, p2: 1.0})
        assert e.is_homogeneous() is None


# ── PathAlgebra ───────────────────────────────────────────────────────

class TestPathAlgebra:
    def test_enumerate_paths_0(self, pa):
        paths = pa.enumerate_paths(0)
        assert len(paths) == 4  # 4 trivial paths

    def test_enumerate_paths_1(self, pa):
        paths = pa.enumerate_paths(1)
        assert len(paths) == 5  # 5 arrows

    def test_enumerate_paths_2(self, pa):
        paths = pa.enumerate_paths(2)
        # length-2 walks: α₁·β, α₂·β, γ·δ = 3
        assert len(paths) == 3

    def test_paths_from_to(self, pa):
        paths = pa.paths_from_to(1, 4, 2)
        assert len(paths) == 3

    def test_dimension(self, pa):
        assert pa.dimension(1, 4, 2) == 3
        assert pa.dimension(1, 2, 1) == 2

    def test_verify_matrix_equivalence(self, pa):
        assert pa.verify_matrix_equivalence() is True

    def test_graded_dimension(self, pa):
        assert pa.graded_dimension(0) == 4
        assert pa.graded_dimension(1) == 5
        assert pa.graded_dimension(2) == 3

    def test_total_dimension(self, pa):
        # 4 + 5 + 3 = 12
        assert pa.total_dimension() == 12

    def test_idempotent(self, pa, q4):
        e1 = pa.idempotent(1)
        assert not e1.is_zero()

    def test_arrow_element(self, pa):
        a = pa.arrow_element("α₁")
        assert not a.is_zero()


# ── Ideal and QuotientAlgebra ─────────────────────────────────────────

class TestIdealAndQuotient:
    def test_ideal_admissible(self, pa, q4):
        # Relation: α₁·β - α₂·β = 0
        p1 = Path(q4, arrows=["α₁", "β"])
        p2 = Path(q4, arrows=["α₂", "β"])
        gen = PathAlgebraElement({p1: 1.0, p2: -1.0})
        I = Ideal(pa, [gen])
        assert I.is_admissible() is True

    def test_ideal_not_admissible(self, pa, q4):
        # Relation with a degree-1 term
        p1 = Path(q4, arrows=["α₁"])
        gen = PathAlgebraElement({p1: 1.0})
        I = Ideal(pa, [gen])
        assert I.is_admissible() is False

    def test_reduction(self, pa, q4):
        p1 = Path(q4, arrows=["α₁", "β"])
        p2 = Path(q4, arrows=["α₂", "β"])
        gen = PathAlgebraElement({p1: 1.0, p2: -1.0})
        I = Ideal(pa, [gen])

        # Reduce α₁·β → should become α₂·β
        elem = PathAlgebraElement({p1: 1.0})
        reduced = I.reduce(elem)
        assert p2 in reduced.terms

    def test_quotient_dimension(self, pa, q4):
        p1 = Path(q4, arrows=["α₁", "β"])
        p2 = Path(q4, arrows=["α₂", "β"])
        gen = PathAlgebraElement({p1: 1.0, p2: -1.0})
        I = Ideal(pa, [gen])
        QA = QuotientAlgebra(pa, I)

        # dim(e_1 · (kQ/I)_2 · e_4) should be 2 (α₂·β and γ·δ, since α₁·β ≡ α₂·β)
        assert QA.dimension(1, 4, 2) == 2

    def test_quotient_total_dimension(self, pa, q4):
        p1 = Path(q4, arrows=["α₁", "β"])
        p2 = Path(q4, arrows=["α₂", "β"])
        gen = PathAlgebraElement({p1: 1.0, p2: -1.0})
        I = Ideal(pa, [gen])
        QA = QuotientAlgebra(pa, I)

        # Total: 4 (deg 0) + 5 (deg 1) + 2 (deg 2) = 11
        assert QA.total_dimension() == 11
