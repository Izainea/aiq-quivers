"""
AIQ — Autómatas de Impacto sobre Quivers
=========================================

Implementación computacional del marco AIQ definido en:
  Zainea Maya & Moreno Cañadas, "Autómatas de Impacto sobre Quivers:
  Un Marco Algebraico para Modelar Dinámicas de Sistemas Complejos"

Módulos:
  quiver        — Quiver Q=(Q0,Q1,s,t), matriz de adyacencia
  path_algebra  — Álgebra de caminos kQ, ideales, cociente kQ/I
  impact        — Grado/vector de impacto, SFV, tasas de impacto
  automaton     — Clase AIQ, reglas SI/SIS/SIR, órbitas
  morphisms     — Morfismos de quivers y de AIQs
  visualization — Dibujo de quivers, heatmaps, evolución
  datasets      — Ejemplos del paper + datasets Cora y cit-HepPh
  validation    — Validación temporal con datos de citación reales
  brauer        — Configuraciones de Brauer y BCA (Green-Schroll, Cañadas et al.)
"""

from .quiver import Quiver
from .path_algebra import Path, PathAlgebraElement, PathAlgebra, Ideal, QuotientAlgebra
from .impact import (
    impact_degree, impact_vector, impact_vector_matrix,
    FundamentalNeighborhoodSystem,
    impact_rate_simple, impact_rate_enriched, impact_rate_signed,
)
from .automaton import AIQ
from .brauer import (
    BrauerConfiguration, brauer_from_citation_json,
    mu_standard, mu_uniform, mu_from_data, MU_STRATEGIES,
)
from . import morphisms
from . import visualization
from . import datasets
from . import validation
from . import brauer

__version__ = "1.1.0"
