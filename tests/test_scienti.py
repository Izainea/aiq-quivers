"""
Tests para aiq.scienti — cargadores de CvLAC y GrupLAC del scraper Scienti.

Los tests usan un directorio temporal con JSONs sintéticos que reproducen
el formato emitido por `scraper_scienti`. Si existe el directorio real
``../scraper_scienti/data`` se ejecutan también pruebas humo contra él.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiq import scienti
from aiq.brauer import BrauerConfiguration
from aiq.quiver import Quiver


# ── Fixtures ──────────────────────────────────────────────

CVLAC_SAMPLE_A = {
    "cod_rh": "0000000001",
    "nombre": "GARCÍA LÓPEZ, ANA",
    "categoria_minciencias": "Senior",
    "nivel_maximo": "Doctorado",
    "lineas_investigacion": [
        "Álgebra,",
        "Teoría de representaciones,",
    ],
    "produccion": {
        "articulos": {
            "total": 2,
            "items": [
                'GARCIA LOPEZ ANA, PEREZ MARIA, "Sobre álgebras de Brauer". Rev Mat 2020',
                'GARCIA LOPEZ ANA, RUIZ JUAN, "Quivers y caminos". Rev Mat 2022',
            ],
        }
    },
}

CVLAC_SAMPLE_B = {
    "cod_rh": "0000000002",
    "nombre": "PEREZ, MARIA",
    "categoria_minciencias": "Asociado",
    "nivel_maximo": "Doctorado",
    "lineas_investigacion": ["Álgebra,"],
    "produccion": {
        "articulos": {
            "total": 1,
            "items": [
                'GARCIA LOPEZ ANA, PEREZ MARIA, "Sobre álgebras de Brauer". Rev Mat 2020',
            ],
        }
    },
}

GRUPLAC_SAMPLE = {
    "nro_gruplac": "00000000000001",
    "nombre": "Grupo Álgebras",
    "lider": "GARCIA LOPEZ ANA",
    "clasificacion": "A1",
    "departamento_ciudad": "BOGOTÁ",
    "area_conocimiento": "Matemáticas",
    "integrantes": [
        {"nombre": "GARCIA LOPEZ ANA", "cod_rh": "0000000001"},
        {"nombre": "PEREZ, MARIA", "cod_rh": "0000000002"},
    ],
}


@pytest.fixture
def scienti_root(tmp_path: Path) -> Path:
    """Directorio raíz emulando ``scraper_scienti/data``."""
    (tmp_path / "cvlac").mkdir()
    (tmp_path / "gruplac").mkdir()

    for rec in (CVLAC_SAMPLE_A, CVLAC_SAMPLE_B):
        with open(tmp_path / "cvlac" / f"{rec['cod_rh']}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)

    with open(
        tmp_path / "gruplac" / f"{GRUPLAC_SAMPLE['nro_gruplac']}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(GRUPLAC_SAMPLE, f, ensure_ascii=False)

    return tmp_path


# ── Carga cruda ──────────────────────────────────────────

def test_load_cvlac_records(scienti_root):
    records = scienti.load_cvlac_records(root=scienti_root)
    assert set(records) == {"0000000001", "0000000002"}
    assert records["0000000001"]["nombre"] == "GARCÍA LÓPEZ, ANA"


def test_load_gruplac_records(scienti_root):
    records = scienti.load_gruplac_records(root=scienti_root)
    assert "00000000000001" in records
    assert records["00000000000001"]["lider"] == "GARCIA LOPEZ ANA"


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scienti.load_cvlac_records(root=tmp_path / "no_existe")


# ── Quivers ──────────────────────────────────────────────

def test_coauthorship_quiver(scienti_root):
    q, meta = scienti.load_coauthorship_quiver(root=scienti_root)
    assert isinstance(q, Quiver)
    # Tres autores deberían detectarse: García López Ana, Pérez María, Ruiz Juan
    names = set(q.Q0)
    assert any("GARCIA" in n and "ANA" in n for n in names)
    assert any("PEREZ" in n for n in names)
    # Las flechas son simétricas (a→b y b→a)
    assert len(q.Q1) % 2 == 0
    # Cada autor extraído tiene contador de artículos > 0
    assert all(info["articulos"] > 0 for info in meta.values())


def test_coauthorship_quiver_min_articles_filter(scienti_root):
    q, _ = scienti.load_coauthorship_quiver(root=scienti_root, min_articles=2)
    # Sólo Ana queda: aparece en 2 artículos
    assert any("ANA" in n for n in q.Q0)
    assert not any("RUIZ" in n for n in q.Q0)


def test_group_member_quiver(scienti_root):
    q, meta = scienti.load_group_member_quiver(root=scienti_root)
    grupo = "grp:00000000000001"
    assert grupo in q.Q0
    assert "rh:0000000001" in q.Q0
    assert "rh:0000000002" in q.Q0
    # 2 aristas grupo → miembro
    assert len(q.Q1) == 2
    assert meta[grupo]["clasificacion"] == "A1"


def test_researcher_group_quiver_bidirectional(scienti_root):
    q, _ = scienti.load_researcher_group_quiver(root=scienti_root)
    # Cada arista grupo→miembro tiene su recíproca
    assert len(q.Q1) == 4
    forward = {(s, t) for _, s, t in q.Q1 if str(s).startswith("grp:")}
    backward = {(s, t) for _, s, t in q.Q1 if str(t).startswith("grp:")}
    assert {(t, s) for s, t in forward} == backward


def test_research_line_quiver(scienti_root):
    q, meta = scienti.load_research_line_quiver(root=scienti_root)
    assert any(str(v).startswith("line:álgebra") for v in q.Q0)
    # Ana declara 2 líneas, María declara 1 → 3 aristas
    assert len(q.Q1) == 3


# ── Brauer ───────────────────────────────────────────────

def test_brauer_from_scienti(scienti_root):
    bc, pdata = scienti.load_scienti_brauer_config(root=scienti_root, min_authors=2)
    assert isinstance(bc, BrauerConfiguration)
    # Tres artículos distintos en la fixture, todos con ≥2 coautores
    assert len(bc._polygons) >= 2
    # Todos los polígonos tienen año asignado (2020 o 2022)
    years = {p["year"] for p in pdata.values()}
    assert years.issubset({2020, 2022})


# ── Smoke test contra datos reales si están disponibles ──

_REAL_ROOT = Path(__file__).parent.parent.parent / "scraper_scienti" / "data"


@pytest.mark.skipif(
    not (_REAL_ROOT / "cvlac").exists(),
    reason="Datos reales del scraper Scienti no disponibles",
)
def test_smoke_real_cvlac():
    records = scienti.load_cvlac_records(limit=5)
    assert len(records) > 0
    q, meta = scienti.load_coauthorship_quiver(limit=10)
    assert len(q.Q0) > 0
