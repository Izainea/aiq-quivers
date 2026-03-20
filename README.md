# AIQ — Impact Automata on Quivers

[![PyPI version](https://img.shields.io/pypi/v/aiq-quivers.svg)](https://pypi.org/project/aiq-quivers/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An algebraic framework for modeling dynamics on complex systems using quiver
representations, path algebras, and Brauer configuration algebras.

Based on: **Zainea Maya & Moreno Cañadas**, *"Autómatas de Impacto sobre Quivers:
Un Marco Algebraico para Modelar Dinámicas de Sistemas Complejos"*.

## Features

| Module | Description |
|--------|-------------|
| `quiver` | Quiver Q=(Q₀,Q₁,s,t) with adjacency matrices, BFS distances, O(1) degree queries |
| `path_algebra` | Path algebras kQ, ideals I, quotient algebras kQ/I |
| `impact` | Impact degree, vector, Fundamental Neighborhood System (SFV), impact rates |
| `automaton` | Impact automata (AIQ) with SI/SIS/SIR evolution rules, Monte Carlo |
| `brauer` | Brauer configuration algebras: δ_B, entropy H(B), dimension formulas |
| `morphisms` | Morphisms of quivers and AIQs (base, dynamic, algebraic) |
| `datasets` | Paper examples + Cora + cit-HepPh citation networks |
| `validation` | Temporal validation with real citation data |
| `visualization` | Quiver drawing, heatmaps, evolution grids, validation dashboard |

## Installation

```bash
pip install aiq-quivers
```

For development:
```bash
git clone https://github.com/izaineam/aiq-quivers.git
cd aiq-quivers
pip install -e ".[dev]"
```

## Quick Start

### Impact Automaton (AIQ)

```python
from aiq import Quiver, AIQ

# Define a citation quiver
Q = Quiver(
    vertices=["P1", "P2", "P3", "P4", "P5"],
    arrows=[
        ("a1", "P1", "P2"), ("a2", "P1", "P3"),
        ("a3", "P2", "P4"), ("a4", "P3", "P4"),
        ("a5", "P3", "P5"), ("a6", "P4", "P5"),
    ],
)

# Create AIQ with SIR rule
aiq = AIQ(Q, states=["S", "R", "O"], evolution_rule="SIR",
          beta=1.0, recovery_prob=0.3)

# Set initial config: P1 is Relevant, rest Susceptible
aiq.set_initial_config({"P1": "R", "P2": "S", "P3": "S", "P4": "S", "P5": "S"})

# Run 10 steps
aiq.run(10, seed=42)
print(aiq.state_counts())
```

### Brauer Configuration Analysis

```python
from aiq.brauer import example_partitions_of_10

bc = example_partitions_of_10()
print(bc.summary())
# BrauerConfiguration(|Γ₀|=4, |Γ₁|=3, δ_B=13, H(B)=1.8339)
#   dim_k Λ_M = 45
#   dim_k Z(Λ_M) = 14
#   δ_B = 13
#   H(B) = 1.833927

# From citation network JSON
from aiq.brauer import brauer_from_citation_json
bc = brauer_from_citation_json("data/canadas_citation_network.json")
analysis = bc.brauer_analysis()
```

### Path Algebra

```python
from aiq import Quiver, PathAlgebra

Q = Quiver([1, 2, 3, 4],
           [("α₁", 1, 2), ("α₂", 1, 2), ("γ", 1, 3), ("β", 2, 4), ("δ", 3, 4)])

pa = PathAlgebra(Q)
assert pa.verify_matrix_equivalence()  # dim(e_i·kQ_k·e_j) == (A^k)_{ij}
print(f"dim(kQ) = {pa.total_dimension()}")  # 12
```

## Key Concepts

- **Impact degree** g(cᵢ, cⱼ): shortest directed path length (Def. 2.2)
- **Impact vector** vec_g(cᵢ, cⱼ): walk counts by length (Def. 2.5)
- **SFV** {A_k(c)}: fundamental neighborhood system (Def. 2.3)
- **Impact rate** i^t(c): weighted influence from neighbors (Def. 2.9)
- **Brauer factor** δ_B = Σ μ(m)·val(m): algebraic impact invariant
- **Brauer entropy** H(B): Shannon entropy of the configuration

## Running Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@article{zainea2025aiq,
  title={Aut{\'o}matas de Impacto sobre Quivers:
         Un Marco Algebraico para Modelar Din{\'a}micas de Sistemas Complejos},
  author={Zainea Maya, Isaac and Moreno Ca{\~n}adas, Agust{\'i}n},
  year={2025},
  institution={Universidad Nacional de Colombia}
}
```

## License

MIT License. See [LICENSE](LICENSE).
