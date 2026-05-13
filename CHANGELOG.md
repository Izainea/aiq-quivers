# Changelog

All notable changes to `aiq-quivers` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.2.0]: https://github.com/Izainea/aiq-quivers/releases/tag/v1.2.0
[1.1.1]: https://github.com/Izainea/aiq-quivers/releases/tag/v1.1.1
