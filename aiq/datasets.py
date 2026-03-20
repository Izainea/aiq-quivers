"""
Datasets y ejemplos del paper.

Referencia: ACT.tex, Ejemplos 2.3, 2.7, 3.1, 4.1, 4.5.
Dataset Cora: McCallum et al. (2000).
Dataset cit-HepPh: SNAP (Leskovec et al.).
"""

from __future__ import annotations

import gzip
import os
import urllib.request
import tarfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .quiver import Quiver


# ═══════════════════════════════════════════════════════════════════════
# Ejemplos del paper
# ═══════════════════════════════════════════════════════════════════════

def example_4node_quiver() -> Quiver:
    """
    Quiver de 4 nodos del Ejemplo 2.3.

    Q_0 = {1, 2, 3, 4}
    Q_1 = {α₁:1→2, α₂:1→2, γ:1→3, β:2→4, δ:3→4}

    Dos flechas de 1 a 2 (flechas múltiples).
    Diámetro = 2, es un DAG.
    """
    vertices = [1, 2, 3, 4]
    arrows = [
        ("α₁", 1, 2),
        ("α₂", 1, 2),
        ("γ", 1, 3),
        ("β", 2, 4),
        ("δ", 3, 4),
    ]
    return Quiver(vertices, arrows)


def example_4node_initial_config() -> dict:
    """
    Configuración inicial del Ejemplo 2.7 y Ejemplo 3.1.

    π⁰ = {1: I, 2: S, 3: S, 4: S}
    Solo el vértice 1 está infectado/relevante.
    """
    return {1: "I", 2: "S", 3: "S", 4: "S"}


def example_5node_citation_quiver() -> Quiver:
    """
    Quiver de citación de 5 artículos del Ejemplo 4.1.

    P1 (artículo originador) cita a P2, P3.
    P2 cita a P4.
    P3 cita a P4, P5.
    P4 cita a P5.

    Modela obsolescencia del artículo originador.
    """
    vertices = ["P1", "P2", "P3", "P4", "P5"]
    arrows = [
        ("a1", "P1", "P2"),
        ("a2", "P1", "P3"),
        ("a3", "P2", "P4"),
        ("a4", "P3", "P4"),
        ("a5", "P3", "P5"),
        ("a6", "P4", "P5"),
    ]
    return Quiver(vertices, arrows)


def example_5node_citation_config() -> dict:
    """
    Configuración inicial Ejemplo 4.1.
    P1 = R (relevante/citado), resto S (no citado aún).
    """
    return {"P1": "R", "P2": "S", "P3": "S", "P4": "S", "P5": "S"}


def example_gene_regulatory_quiver() -> Quiver:
    """
    Red de regulación génica del Ejemplo 4.5 (con signos).

    Genes G1, G2, G3 con:
    - G1 → G2 (activación, +1)
    - G1 → G3 (inhibición, -1)
    - G2 → G3 (activación, +1)

    Muestra el efecto de signos en las tasas de impacto.
    """
    vertices = ["G1", "G2", "G3"]
    arrows = [
        ("act_12", "G1", "G2"),
        ("inh_13", "G1", "G3"),
        ("act_23", "G2", "G3"),
    ]
    weights = {
        "act_12": +1,
        "inh_13": -1,
        "act_23": +1,
    }
    return Quiver(vertices, arrows, weights)


def example_supply_chain_quiver() -> Quiver:
    """
    Cadena de suministro simplificada (Ejemplo 4.3).

    Proveedor → Fábrica → Distribuidor → Tienda
    con ruta alternativa: Proveedor → Fábrica_B → Distribuidor
    """
    vertices = ["Proveedor", "Fábrica_A", "Fábrica_B", "Distribuidor", "Tienda"]
    arrows = [
        ("s_fa", "Proveedor", "Fábrica_A"),
        ("s_fb", "Proveedor", "Fábrica_B"),
        ("fa_d", "Fábrica_A", "Distribuidor"),
        ("fb_d", "Fábrica_B", "Distribuidor"),
        ("d_t", "Distribuidor", "Tienda"),
    ]
    return Quiver(vertices, arrows)


def example_disinformation_quiver() -> Quiver:
    """
    Red de desinformación simplificada (Ejemplo 4.4).

    Fuente → Amplificador_1, Amplificador_2 → Audiencia_1, Audiencia_2, Audiencia_3
    con un VerificadorFact que inhibe la propagación.
    """
    vertices = ["Fuente", "Amp1", "Amp2", "Aud1", "Aud2", "Aud3", "Fact"]
    arrows = [
        ("f_a1", "Fuente", "Amp1"),
        ("f_a2", "Fuente", "Amp2"),
        ("a1_u1", "Amp1", "Aud1"),
        ("a1_u2", "Amp1", "Aud2"),
        ("a2_u2", "Amp2", "Aud2"),
        ("a2_u3", "Amp2", "Aud3"),
        ("fact_u1", "Fact", "Aud1"),
        ("fact_u2", "Fact", "Aud2"),
    ]
    weights = {
        "f_a1": +1, "f_a2": +1,
        "a1_u1": +1, "a1_u2": +1,
        "a2_u2": +1, "a2_u3": +1,
        "fact_u1": -1,  # Verificador inhibe
        "fact_u2": -1,
    }
    return Quiver(vertices, arrows, weights)


def example_urban_mobility_quiver() -> Quiver:
    """
    Red de movilidad urbana simplificada (Ejemplo 4.2).

    Zonas de una ciudad conectadas por rutas de transporte.
    """
    vertices = ["Centro", "Norte", "Sur", "Este", "Oeste", "Aeropuerto"]
    arrows = [
        ("cn", "Centro", "Norte"),
        ("nc", "Norte", "Centro"),
        ("cs", "Centro", "Sur"),
        ("sc", "Sur", "Centro"),
        ("ce", "Centro", "Este"),
        ("ec", "Este", "Centro"),
        ("co", "Centro", "Oeste"),
        ("oc", "Oeste", "Centro"),
        ("ea", "Este", "Aeropuerto"),
        ("ae", "Aeropuerto", "Este"),
        ("ns", "Norte", "Sur"),
    ]
    return Quiver(vertices, arrows)


# ═══════════════════════════════════════════════════════════════════════
# Dataset Cora
# ═══════════════════════════════════════════════════════════════════════

_CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
_CORA_DIR = Path(__file__).parent.parent / "data" / "cora"


def _download_cora(data_dir: Optional[Path] = None) -> Path:
    """Descarga y extrae el dataset Cora si no existe."""
    data_dir = data_dir or _CORA_DIR
    data_dir = Path(data_dir)

    cites_file = data_dir / "cora.cites"
    content_file = data_dir / "cora.content"

    if cites_file.exists() and content_file.exists():
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = data_dir / "cora.tgz"

    print(f"Descargando Cora desde {_CORA_URL}...")
    urllib.request.urlretrieve(_CORA_URL, tgz_path)

    print("Extrayendo...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")

    # Los archivos suelen estar en cora/cora.cites
    nested = data_dir / "cora"
    if nested.exists() and nested.is_dir():
        for f in nested.iterdir():
            f.rename(data_dir / f.name)
        nested.rmdir()

    if tgz_path.exists():
        tgz_path.unlink()

    print(f"Cora descargado en {data_dir}")
    return data_dir


def load_cora(
    data_dir: Optional[str] = None,
) -> tuple:
    """
    Carga el dataset Cora como un Quiver + metadatos.

    Returns
    -------
    quiver : Quiver
        Grafo de citación (2,708 nodos, ~5,429 aristas).
    metadata : dict
        {paper_id: {'label': str, 'features': np.array}}
    """
    d = _download_cora(Path(data_dir) if data_dir else None)

    # Leer contenido: paper_id word1 word2 ... label
    content_file = d / "cora.content"
    metadata = {}
    paper_ids = []

    with open(content_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            pid = parts[0]
            features = np.array([int(x) for x in parts[1:-1]], dtype=np.int8)
            label = parts[-1]
            metadata[pid] = {"label": label, "features": features}
            paper_ids.append(pid)

    paper_set = set(paper_ids)

    # Leer citas: cited citing (o citing cited)
    cites_file = d / "cora.cites"
    arrows = []
    counter = 0
    with open(cites_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            cited, citing = parts[0], parts[1]
            # Flecha: citing → cited (citar es dirigir influencia hacia el citado)
            if citing in paper_set and cited in paper_set:
                arrows.append((f"c_{counter}", citing, cited))
                counter += 1

    quiver = Quiver(paper_ids, arrows)
    return quiver, metadata


def load_cora_subset(
    n: int = 200,
    seed: int = 42,
    data_dir: Optional[str] = None,
) -> tuple:
    """
    Carga un subconjunto conexo de Cora para demos rápidas.

    Estrategia: BFS desde un nodo con alto grado hasta tener n nodos.

    Returns
    -------
    quiver : Quiver
        Sub-quiver con ~n nodos.
    metadata : dict
        Metadatos del subconjunto.
    """
    full_quiver, full_meta = load_cora(data_dir)

    # Encontrar nodo con mayor out-degree como semilla
    rng = np.random.default_rng(seed)
    degrees = {}
    for _, src, _ in full_quiver.Q1:
        degrees[src] = degrees.get(src, 0) + 1

    if not degrees:
        return full_quiver, full_meta

    # Top 10 nodos por grado, elegir uno aleatoriamente
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]
    start = sorted_nodes[rng.integers(0, len(sorted_nodes))]

    # BFS para obtener subconjunto conexo
    from collections import deque
    visited = {start}
    queue = deque([start])

    # Construir lista de adyacencia (ambas direcciones para conexidad)
    adj = {}
    for _, src, tgt in full_quiver.Q1:
        adj.setdefault(src, []).append(tgt)
        adj.setdefault(tgt, []).append(src)

    while queue and len(visited) < n:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append(v)
                if len(visited) >= n:
                    break

    subset = list(visited)
    sub_quiver = full_quiver.subquiver(subset)
    sub_meta = {pid: full_meta[pid] for pid in subset if pid in full_meta}

    return sub_quiver, sub_meta


# ═══════════════════════════════════════════════════════════════════════
# Dataset cit-HepPh (SNAP) — con fechas de publicación
# ═══════════════════════════════════════════════════════════════════════

_HEPPH_EDGES_URL = "https://snap.stanford.edu/data/cit-HepPh.txt.gz"
_HEPPH_DATES_URL = "https://snap.stanford.edu/data/cit-HepPh-dates.txt.gz"
_HEPPH_DIR = Path(__file__).parent.parent / "data" / "cit-hepph"


def _download_hepph(data_dir: Optional[Path] = None) -> Path:
    """Descarga y descomprime el dataset cit-HepPh si no existe."""
    data_dir = data_dir or _HEPPH_DIR
    data_dir = Path(data_dir)

    edges_file = data_dir / "cit-HepPh.txt"
    dates_file = data_dir / "cit-HepPh-dates.txt"

    if edges_file.exists() and dates_file.exists():
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    for url, out_name in [
        (_HEPPH_EDGES_URL, "cit-HepPh.txt"),
        (_HEPPH_DATES_URL, "cit-HepPh-dates.txt"),
    ]:
        gz_path = data_dir / (out_name + ".gz")
        out_path = data_dir / out_name

        if not out_path.exists():
            print(f"Descargando {url}...")
            urllib.request.urlretrieve(url, gz_path)
            print(f"Descomprimiendo {out_name}...")
            with gzip.open(gz_path, "rb") as f_in:
                with open(out_path, "wb") as f_out:
                    f_out.write(f_in.read())
            gz_path.unlink()

    print(f"cit-HepPh descargado en {data_dir}")
    return data_dir


def load_hepph(
    data_dir: Optional[str] = None,
) -> tuple:
    """
    Carga el dataset cit-HepPh (SNAP) como un Quiver + metadatos temporales.

    Este dataset contiene citaciones de artículos de Física de Altas Energías
    (arXiv hep-ph) con fechas exactas de publicación (1992–2003).

    Returns
    -------
    quiver : Quiver
        Grafo de citación (~34,500 nodos, ~420,000 aristas).
    metadata : dict
        {paper_id: {'date': datetime.date, 'year': int}}
    """
    d = _download_hepph(Path(data_dir) if data_dir else None)

    # 1. Parsear fechas
    dates_map = {}
    dates_file = d / "cit-HepPh-dates.txt"
    with open(dates_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            # Formato: paper_id\tdate
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            pid = parts[0].strip()
            date_str = parts[1].strip()
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                dates_map[pid] = {"date": dt, "year": dt.year}
            except ValueError:
                continue

    # 2. Parsear aristas
    edges_file = d / "cit-HepPh.txt"
    arrows = []
    counter = 0
    skipped_temporal = 0

    with open(edges_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            # Solo incluir si ambos tienen fecha
            if src not in dates_map or tgt not in dates_map:
                continue
            # Convención: src cita a tgt → flecha src → tgt
            # (el citante dirige influencia hacia el citado)
            arrows.append((f"c_{counter}", src, tgt))
            counter += 1

    # Construir lista de vértices (solo los que participan en aristas)
    paper_set = set()
    for _, s, t in arrows:
        paper_set.add(s)
        paper_set.add(t)
    paper_ids = sorted(paper_set)
    metadata = {pid: dates_map[pid] for pid in paper_ids}

    quiver = Quiver(paper_ids, arrows)
    return quiver, metadata


def load_hepph_subset(
    n: int = 500,
    seed: int = 42,
    year_range: Optional[tuple] = None,
    data_dir: Optional[str] = None,
) -> tuple:
    """
    Carga un subconjunto conexo de cit-HepPh para simulaciones AIQ.

    Parameters
    ----------
    n : int
        Número aproximado de nodos.
    year_range : tuple or None
        (año_min, año_max) para filtrar papers por fecha de publicación.
        Ejemplo: (1995, 1999) para papers de 1995 a 1999.

    Returns
    -------
    quiver : Quiver
        Sub-quiver con ~n nodos.
    metadata : dict
        Metadatos del subconjunto con fechas.
    """
    full_quiver, full_meta = load_hepph(data_dir)

    # Filtrar por rango de años si se especifica
    if year_range is not None:
        y_min, y_max = year_range
        valid_papers = {
            pid for pid, m in full_meta.items()
            if y_min <= m["year"] <= y_max
        }
        # Re-filtrar aristas
        filtered_arrows = [
            (name, s, t) for name, s, t in full_quiver.Q1
            if s in valid_papers and t in valid_papers
        ]
        # Vértices que participan
        filtered_verts = set()
        for _, s, t in filtered_arrows:
            filtered_verts.add(s)
            filtered_verts.add(t)
        filtered_verts = sorted(filtered_verts)
        if not filtered_verts:
            raise ValueError(f"No hay papers en el rango {year_range}")
        full_quiver = Quiver(filtered_verts, filtered_arrows)
        full_meta = {pid: full_meta[pid] for pid in filtered_verts}

    # BFS desde nodo de alto grado
    rng = np.random.default_rng(seed)
    degrees = {}
    for _, src, _ in full_quiver.Q1:
        degrees[src] = degrees.get(src, 0) + 1

    if not degrees:
        return full_quiver, full_meta

    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]
    start = sorted_nodes[rng.integers(0, len(sorted_nodes))]

    visited = {start}
    queue = deque([start])

    adj = {}
    for _, src, tgt in full_quiver.Q1:
        adj.setdefault(src, []).append(tgt)
        adj.setdefault(tgt, []).append(src)

    while queue and len(visited) < n:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append(v)
                if len(visited) >= n:
                    break

    subset = sorted(visited)
    sub_quiver = full_quiver.subquiver(subset)
    sub_meta = {pid: full_meta[pid] for pid in subset if pid in full_meta}

    return sub_quiver, sub_meta


# ═══════════════════════════════════════════════════════════════════════
# Dataset de citaciones de Cañadas para análisis de Brauer
# ═══════════════════════════════════════════════════════════════════════

def load_canadas_citation_network():
    """
    Cargar la red de citaciones de Agustín Moreno Cañadas et al.
    para análisis de configuraciones de Brauer.

    Returns
    -------
    BrauerConfiguration
        Configuración de Brauer donde:
        - Γ₀ = referencias (papers citados)
        - Γ₁ = 17 artículos de Cañadas (polígonos/multiconjuntos)
        - O = orden por año de publicación (Pwords ordenados cronológicamente)

    dict
        Datos crudos del JSON (papers, reference_pool, metadata).
    """
    from .brauer import brauer_from_citation_json
    import json

    json_path = Path(__file__).parent.parent / "data" / "canadas_citation_network.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Archivo de datos no encontrado: {json_path}\n"
            "Ejecute desde el directorio quivers_analysis/."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    bc = brauer_from_citation_json(json_path)
    return bc, raw_data
