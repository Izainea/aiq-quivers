"""
Cargadores para datos del scraper Scienti (Minciencias).

Lee los JSON producidos por `scraper_scienti` (CvLAC e GrupLAC) y los
convierte en quivers para análisis con AIQ y configuraciones de Brauer.

Estructura esperada del directorio de datos (configurable):

    <root>/
    ├── cvlac/      # un JSON por investigador, nombre = cod_rh.json
    └── gruplac/    # un JSON por grupo, nombre = nro_gruplac.json

Por defecto se busca ``../scraper_scienti/data`` relativo al paquete
(``quivers_analysis/``), pero cualquier ruta puede pasarse vía argumento.

Quivers expuestos
-----------------
* :func:`load_coauthorship_quiver`  — investigadores ↔ coautoría (no dirigido,
  representado con dos flechas)
* :func:`load_group_member_quiver`  — grupo → integrante
* :func:`load_researcher_group_quiver` — investigador ↔ grupo (bipartito)
* :func:`load_research_line_quiver` — investigador → línea de investigación
* :func:`load_scienti_brauer_config` — configuración de Brauer donde cada
  artículo es un polígono (multiconjunto de coautores)

Las funciones devuelven, además del quiver, un ``dict`` de metadatos por
vértice (nombre, nivel, categoría Minciencias, año, etc.).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .quiver import Quiver


# ═══════════════════════════════════════════════════════════════════════
# Configuración de rutas
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_SCIENTI_ROOT = (
    Path(__file__).parent.parent.parent / "scraper_scienti" / "data"
)


def _resolve_root(root: Optional[str | Path]) -> Path:
    """Resuelve y valida el directorio raíz de datos Scienti."""
    p = Path(root) if root else _DEFAULT_SCIENTI_ROOT
    if not p.exists():
        raise FileNotFoundError(
            f"Directorio Scienti no encontrado: {p}\n"
            "Pase `root=` apuntando al directorio `data/` del scraper."
        )
    return p


def _iter_json(directory: Path, limit: Optional[int] = None) -> Iterable[Path]:
    """Itera archivos .json en un directorio, opcionalmente acotado."""
    if not directory.exists():
        return
    files = sorted(directory.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    yield from files


def _load_json(path: Path) -> Optional[dict]:
    """Lee un JSON tolerando archivos vacíos o corruptos."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ═══════════════════════════════════════════════════════════════════════
# Carga cruda
# ═══════════════════════════════════════════════════════════════════════

def load_cvlac_records(
    root: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> dict[str, dict]:
    """
    Carga todos los CvLAC disponibles.

    Returns
    -------
    dict
        ``{cod_rh: registro_cvlac}``
    """
    base = _resolve_root(root) / "cvlac"
    out: dict[str, dict] = {}
    for path in _iter_json(base, limit):
        data = _load_json(path)
        if not data:
            continue
        cod = str(data.get("cod_rh", path.stem)).zfill(10)
        out[cod] = data
    return out


def load_gruplac_records(
    root: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> dict[str, dict]:
    """
    Carga todos los GrupLAC disponibles.

    Returns
    -------
    dict
        ``{nro_gruplac: registro_gruplac}``
    """
    base = _resolve_root(root) / "gruplac"
    out: dict[str, dict] = {}
    for path in _iter_json(base, limit):
        data = _load_json(path)
        if not data:
            continue
        nro = str(data.get("nro_gruplac", path.stem)).zfill(14)
        out[nro] = data
    return out


# ═══════════════════════════════════════════════════════════════════════
# Helpers de extracción
# ═══════════════════════════════════════════════════════════════════════

# Nombres en CvLAC suelen aparecer en MAYÚSCULAS, separados por comas.
# Heurística: una secuencia de 2+ palabras totalmente en mayúsculas.
_AUTHOR_RE = re.compile(
    r"\b([A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ'\-]+(?:\s+[A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ'\-]+){1,4})\b"
)


def _normalize_name(name: str) -> str:
    """Normaliza un nombre para usarlo como vértice estable."""
    name = name.strip().upper()
    name = re.sub(r"\s+", " ", name)
    return name


def _extract_authors(item: str) -> list[str]:
    """Extrae nombres de autor de un string de producción CvLAC."""
    # Tomar el prefijo antes del primer paréntesis o comilla — ahí suelen
    # estar listados los autores separados por comas.
    head = re.split(r'["“(]', item, maxsplit=1)[0]
    candidates = _AUTHOR_RE.findall(head)
    seen, authors = set(), []
    for c in candidates:
        n = _normalize_name(c)
        if len(n) < 5 or n in seen:
            continue
        seen.add(n)
        authors.append(n)
    return authors


def _articles_of(record: dict) -> list[str]:
    """Devuelve la lista cruda de artículos de un CvLAC."""
    prod = record.get("produccion", {}) or {}
    art = prod.get("articulos", {}) or {}
    return list(art.get("items", []) or [])


def _year_in(text: str) -> Optional[int]:
    """Extrae el primer año plausible (1900–2099) de un string."""
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return int(m.group(1)) if m else None


# ═══════════════════════════════════════════════════════════════════════
# Quivers
# ═══════════════════════════════════════════════════════════════════════

def load_coauthorship_quiver(
    root: Optional[str | Path] = None,
    limit: Optional[int] = None,
    min_articles: int = 1,
) -> tuple[Quiver, dict]:
    """
    Construye el quiver de coautoría a partir de los CvLAC.

    Cada vértice es un autor (nombre normalizado en MAYÚSCULAS). Por cada
    par de coautores en un mismo artículo se añaden dos flechas (a→b y
    b→a), modelando una relación simétrica dentro del formalismo de
    quivers dirigidos.

    Parameters
    ----------
    root : str | Path, optional
        Raíz del directorio ``scraper_scienti/data``.
    limit : int, optional
        Cota de archivos a leer (útil para debugging).
    min_articles : int
        Sólo se incluyen autores con al menos este número de artículos
        detectados (por defecto 1).

    Returns
    -------
    quiver : Quiver
    metadata : dict
        ``{autor: {'cod_rh': str|None, 'articulos': int, 'cvlac_nombre': str|None}}``
    """
    cvlacs = load_cvlac_records(root, limit=limit)

    art_count: dict[str, int] = defaultdict(int)
    cod_for: dict[str, str] = {}
    pairs: dict[tuple[str, str], int] = defaultdict(int)

    for cod_rh, rec in cvlacs.items():
        owner = _normalize_name(rec.get("nombre", "") or "")
        if owner:
            cod_for.setdefault(owner, cod_rh)

        for item in _articles_of(rec):
            authors = _extract_authors(item)
            if owner and owner not in authors:
                authors.append(owner)
            for a in authors:
                art_count[a] += 1
            for i, a in enumerate(authors):
                for b in authors[i + 1:]:
                    if a == b:
                        continue
                    key = (a, b) if a < b else (b, a)
                    pairs[key] += 1

    keep = {a for a, n in art_count.items() if n >= min_articles}
    vertices = sorted(keep)

    arrows = []
    counter = 0
    for (a, b), w in pairs.items():
        if a not in keep or b not in keep:
            continue
        arrows.append((f"co_{counter}", a, b))
        counter += 1
        arrows.append((f"co_{counter}", b, a))
        counter += 1

    metadata = {
        a: {
            "cod_rh": cod_for.get(a),
            "articulos": art_count[a],
            "cvlac_nombre": cvlacs.get(cod_for.get(a, ""), {}).get("nombre"),
        }
        for a in vertices
    }
    return Quiver(vertices, arrows), metadata


def load_group_member_quiver(
    root: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> tuple[Quiver, dict]:
    """
    Quiver dirigido grupo → integrante a partir de GrupLAC.

    Vértices: ``grp:<nro>`` para grupos, ``rh:<cod_rh>`` para investigadores.
    Cada arista representa pertenencia de un investigador a un grupo.

    Returns
    -------
    quiver : Quiver
    metadata : dict
        Atributos por vértice (tipo, nombre, líder, clasificación, etc.).
    """
    groups = load_gruplac_records(root, limit=limit)

    vertex_meta: dict[str, dict] = {}
    arrows = []
    counter = 0

    for nro, g in groups.items():
        gv = f"grp:{nro}"
        vertex_meta[gv] = {
            "tipo": "grupo",
            "nombre": g.get("nombre") or g.get("lider", ""),
            "lider": g.get("lider", ""),
            "clasificacion": g.get("clasificacion", ""),
            "departamento": g.get("departamento_ciudad", ""),
            "area_conocimiento": g.get("area_conocimiento", ""),
        }
        for member in g.get("integrantes", []) or []:
            cod = member.get("cod_rh")
            if not cod:
                continue
            mv = f"rh:{str(cod).zfill(10)}"
            vertex_meta.setdefault(mv, {
                "tipo": "investigador",
                "nombre": member.get("nombre", ""),
            })
            arrows.append((f"m_{counter}", gv, mv))
            counter += 1

    vertices = sorted(vertex_meta)
    return Quiver(vertices, arrows), vertex_meta


def load_researcher_group_quiver(
    root: Optional[str | Path] = None,
    limit: Optional[int] = None,
    bidirectional: bool = True,
) -> tuple[Quiver, dict]:
    """
    Quiver bipartito investigador ↔ grupo.

    A diferencia de :func:`load_group_member_quiver`, este expresa la
    membresía como una relación simétrica entre las dos clases de
    vértices. Útil para análisis de impacto en cualquier dirección
    (un investigador "influye" a sus grupos y viceversa).

    Returns
    -------
    quiver : Quiver
    metadata : dict
    """
    quiver, meta = load_group_member_quiver(root=root, limit=limit)
    if not bidirectional:
        return quiver, meta

    arrows = list(quiver.Q1)
    counter = len(arrows)
    for _, src, tgt in list(quiver.Q1):
        arrows.append((f"r_{counter}", tgt, src))
        counter += 1
    return Quiver(quiver.Q0, arrows), meta


def load_research_line_quiver(
    root: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> tuple[Quiver, dict]:
    """
    Quiver investigador → línea de investigación (bipartito dirigido).

    Vértices: ``rh:<cod_rh>`` y ``line:<linea_normalizada>``.
    Una flecha por cada línea declarada en el CvLAC del investigador.
    """
    cvlacs = load_cvlac_records(root, limit=limit)

    vertex_meta: dict[str, dict] = {}
    arrows = []
    counter = 0

    for cod, rec in cvlacs.items():
        rv = f"rh:{cod}"
        vertex_meta[rv] = {
            "tipo": "investigador",
            "nombre": rec.get("nombre", ""),
            "categoria": rec.get("categoria_minciencias", ""),
            "nivel_maximo": rec.get("nivel_maximo", ""),
        }
        for line in rec.get("lineas_investigacion", []) or []:
            key = re.sub(r"[^\w\sáéíóúñü]", "", line, flags=re.IGNORECASE)
            key = re.sub(r"\s+", "_", key.strip().lower())
            if not key:
                continue
            lv = f"line:{key}"
            vertex_meta.setdefault(lv, {"tipo": "linea", "nombre": line.strip()})
            arrows.append((f"l_{counter}", rv, lv))
            counter += 1

    return Quiver(sorted(vertex_meta), arrows), vertex_meta


# ═══════════════════════════════════════════════════════════════════════
# Configuración de Brauer desde Scienti
# ═══════════════════════════════════════════════════════════════════════

def load_scienti_brauer_config(
    root: Optional[str | Path] = None,
    limit: Optional[int] = None,
    min_authors: int = 2,
):
    """
    Construye una :class:`BrauerConfiguration` donde:

    * ``Γ₀`` = autores (M = unión de coautores detectados),
    * ``Γ₁`` = artículos (cada uno es un multiconjunto de sus coautores),
    * el orden ``O`` se establece cronológicamente por año detectado en el
      texto del artículo (los que no tengan año van al final, en orden
      lexicográfico).

    Parameters
    ----------
    min_authors : int
        Mínimo de coautores distintos requeridos para incluir un artículo
        como polígono (por defecto 2 — un artículo con un único autor no
        aporta multiplicidad útil para Brauer).
    """
    from .brauer import BrauerConfiguration

    cvlacs = load_cvlac_records(root, limit=limit)

    polygons: list[tuple[str, list[str], Optional[int]]] = []
    seen_keys: set[tuple[str, ...]] = set()

    for cod, rec in cvlacs.items():
        owner = _normalize_name(rec.get("nombre", "") or "")
        for item in _articles_of(rec):
            authors = _extract_authors(item)
            if owner and owner not in authors:
                authors.append(owner)
            uniq = sorted(set(authors))
            if len(uniq) < min_authors:
                continue
            key = tuple(uniq)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            year = _year_in(item)
            pid = f"art_{len(polygons):05d}"
            polygons.append((pid, authors, year))

    polygons.sort(key=lambda p: (p[2] is None, p[2] or 0, p[0]))

    polygon_dict = {pid: authors for pid, authors, _ in polygons}
    polygon_data = {
        pid: {"year": year if year is not None else 0}
        for pid, _, year in polygons
    }

    vertex_set = sorted({a for _, authors, _ in polygons for a in authors})

    bc = BrauerConfiguration(
        vertices=vertex_set,
        polygons=polygon_dict,
        polygon_data=polygon_data,
        validate=False,
    )
    return bc, polygon_data
