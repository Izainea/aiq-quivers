# Changelog

All notable changes to `aiq-quivers` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] — 2026-06-08

Bug-fix release: mathematical-correctness and portability fixes found in a full
review of the package. All 206 tests pass; 4 regression tests added.

### Fixed
- `morphisms.py`: `AIQMorphism.is_algebraic_morphism()` was a stub that always
  returned `True`. It now maps each ideal generator into `kQ'`, reduces it
  modulo `I'`, and checks the result is zero — actually verifying `kf(I) ⊆ I'`.
  Covered by `tests/test_morphisms.py::TestAlgebraicMorphism`.
- `brauer.py`: `center_dimension` used a wrong `C_M` (it required
  `val(m) == 1 and μ(m) > 1`). Per Sierra (2018), `C_M = {m : val(m) = 1}` with
  no `μ` condition; corrected. `__init__` now `setdefault(v, 1)` over all
  vertices so `center_dimension`/`impact_factor`/`entropy` no longer `KeyError`
  on an incomplete custom `mu` dict. `n_loops()` docstring rewritten to match the
  actual cyclic-successor construction. Results now pinned in tests.
- `validation.py`: Windows crash fixed — `mp.get_context("fork")` is tried and
  falls back to the platform default (spawn) on `ValueError`; the bogus
  `mp.context.BaseContext.AuthenticationError` reference is gone and the except
  clause is now `(ValueError, RuntimeError, ImportError, OSError)`. The broad
  `except (ValueError, Exception)` is narrowed to `except ValueError`, and each
  cohort uses `seed + i` (Monte-Carlo noise was correlated across cohorts).
- `automaton.py`: `run()` now threads a single `np.random.Generator` through all
  steps (added an optional `rng=` parameter to `step()`) instead of re-seeding a
  fresh generator per step; `state_counts` raises on states outside `Σ`.
- `quiver.py`: `from_networkx` uses `key is not None` (so MultiDiGraph edge key
  `0` is handled) and dedupes arrow names against collisions.
- `path_algebra.py`: `PathAlgebraElement.__mul__` accepts a numeric scalar, and
  `Ideal.reduce` warns when it hits `max_iter` without converging.

## [1.2.0] — 2026-05-12

### Added
- New module `aiq.scienti` with loaders for the Colombian Scienti
  (Minciencias) data ecosystem. Reads JSONs produced by external
  scrapers of CvLAC (researcher CVs) and GrupLAC (research groups)
  and turns them into:
  - `load_cvlac_records` / `load_gruplac_records` — raw record
    dictionaries keyed by `cod_rh` / `nro_gruplac`.
  - `load_coauthorship_quiver` — coauthorship quiver from CvLAC
    article lists, with author-name normalization and a
    `min_articles` cutoff.
  - `load_group_member_quiver` — directed group → researcher quiver.
  - `load_researcher_group_quiver` — bipartite researcher ↔ group
    quiver (optionally bidirectional).
  - `load_research_line_quiver` — researcher → research-line
    bipartite quiver, normalizing line names.
  - `load_scienti_brauer_config` — `BrauerConfiguration` whose
    polygons are articles (multisets of coauthors), oriented
    chronologically by detected year.
- Re-exports of the new loaders from `aiq.datasets` so they live next
  to the canonical Cora, cit-HepPh and Cañadas loaders.
- `tests/test_scienti.py` covering the new module against synthetic
  fixtures and (when available) real scraped data.

### Changed
- `aiq.__init__` now exposes the `scienti` submodule.

## [1.1.1]

Initial public release on PyPI (status snapshot prior to this changelog).

[1.2.1]: https://github.com/Izainea/aiq-quivers/releases/tag/v1.2.1
[1.2.0]: https://github.com/Izainea/aiq-quivers/releases/tag/v1.2.0
[1.1.1]: https://github.com/Izainea/aiq-quivers/releases/tag/v1.1.1
