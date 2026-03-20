"""
Validación temporal de la teoría AIQ con datos de citación reales.

Compara las predicciones del modelo AIQ (tasas de impacto, obsolescencia)
con la dinámica empírica de citaciones usando datasets con fechas.
"""

from __future__ import annotations

import os
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats

from .quiver import Quiver
from .impact import FundamentalNeighborhoodSystem
from .automaton import AIQ


# ═══════════════════════════════════════════════════════════════════════
# Helpers para paralelismo (top-level → picklable)
# ═══════════════════════════════════════════════════════════════════════

def _default_P(g: int) -> float:
    """P(g) = 1/g  (función con nombre → picklable para multiprocessing)."""
    return 1.0 / g if g >= 1 else 1.0


def _mc_single_run(args: tuple) -> list:
    """Una realización Monte Carlo SIR.  Top-level para ProcessPoolExecutor."""
    (quiver, initial_config, cohort_papers,
     n_steps, beta, alpha, recovery_prob, P_func, g_max, seed) = args

    aiq = AIQ(
        quiver=quiver, states=["S", "R", "O"],
        evolution_rule="SIR",
        beta=beta, alpha=alpha, P=P_func, g_max=g_max,
        impact_mode="simple", recovery_prob=recovery_prob,
    )
    aiq.set_initial_config(initial_config)
    aiq.run(n_steps, seed=seed)

    n_cohort = len(cohort_papers)
    return [
        sum(1 for p in cohort_papers if cfg.get(p) == "R") / n_cohort
        for cfg in aiq.orbit
    ]


# ═══════════════════════════════════════════════════════════════════════
# Métricas empíricas
# ═══════════════════════════════════════════════════════════════════════

def compute_citation_ages(quiver: Quiver, metadata: dict) -> pd.DataFrame:
    """
    Para cada arista (citing → cited), calcula la edad de la citación.

    edad = año_citante - año_citado

    Returns
    -------
    DataFrame con columnas: cited, citing, citation_age_years, citing_year, cited_year
    """
    rows = []
    for _, citing, cited in quiver.Q1:
        if citing in metadata and cited in metadata:
            y_citing = metadata[citing]["year"]
            y_cited = metadata[cited]["year"]
            age = y_citing - y_cited
            rows.append({
                "cited": cited,
                "citing": citing,
                "citation_age_years": age,
                "citing_year": y_citing,
                "cited_year": y_cited,
            })
    return pd.DataFrame(rows)


def compute_empirical_lifetime(quiver: Quiver, metadata: dict) -> pd.DataFrame:
    """
    Para cada paper, calcula su vida útil de citación.

    Returns
    -------
    DataFrame indexado por paper_id con columnas:
        pub_year, first_cited_year, last_cited_year,
        citation_lifetime_years, total_citations, in_degree, out_degree, is_source
    """
    # Construir mapas de in/out con una sola pasada sobre aristas
    incoming = {}   # cited_pid -> list of citing years
    in_deg = {}     # pid -> int
    out_deg = {}    # pid -> int
    for _, citing, cited in quiver.Q1:
        out_deg[citing] = out_deg.get(citing, 0) + 1
        in_deg[cited] = in_deg.get(cited, 0) + 1
        if citing in metadata and cited in metadata:
            incoming.setdefault(cited, []).append(metadata[citing]["year"])

    rows = []
    for pid in quiver.Q0:
        if pid not in metadata:
            continue
        pub_year = metadata[pid]["year"]
        citing_years = incoming.get(pid, [])
        n_in = in_deg.get(pid, 0)

        if citing_years:
            first = min(citing_years)
            last = max(citing_years)
            lifetime = last - pub_year
        else:
            first = None
            last = None
            lifetime = 0

        rows.append({
            "paper_id": pid,
            "pub_year": pub_year,
            "first_cited_year": first,
            "last_cited_year": last,
            "citation_lifetime_years": lifetime,
            "total_citations": len(citing_years),
            "in_degree": n_in,
            "out_degree": out_deg.get(pid, 0),
            "is_source": n_in == 0,
        })

    return pd.DataFrame(rows).set_index("paper_id")


def compute_empirical_decay_curve(
    quiver: Quiver,
    metadata: dict,
    max_age: int = 10,
) -> pd.DataFrame:
    """
    Curva de supervivencia de citaciones.

    Para cada edad a = 0, 1, ..., max_age:
    ¿Qué fracción de papers elegibles sigue recibiendo al menos una citación
    a 'a' años después de su publicación?

    Controla right-censoring: solo cuenta papers publicados ≥ a años
    antes del final del dataset.

    Returns
    -------
    DataFrame con columnas: age, fraction_cited, n_eligible, n_cited
    """
    end_year = max(m["year"] for m in metadata.values())

    # Para cada paper citado, reunir las edades a las que fue citado
    cited_at_age = {}  # pid -> set of ages
    for _, citing, cited in quiver.Q1:
        if citing in metadata and cited in metadata:
            age = metadata[citing]["year"] - metadata[cited]["year"]
            if 0 <= age <= max_age:
                cited_at_age.setdefault(cited, set()).add(age)

    # Pre-computar pub_years como array para vectorizar el conteo de elegibles
    pub_years = np.array([m["year"] for m in metadata.values()])

    rows = []
    for age in range(max_age + 1):
        # Elegibles: papers con pub_year + age <= end_year
        n_eligible = int(np.sum(pub_years + age <= end_year))
        # Citados a esta edad exacta
        n_cited = sum(1 for pid_ages in cited_at_age.values() if age in pid_ages)
        fraction = n_cited / n_eligible if n_eligible > 0 else 0.0
        rows.append({
            "age": age,
            "fraction_cited": fraction,
            "n_eligible": n_eligible,
            "n_cited": n_cited,
        })
    return pd.DataFrame(rows)


def compute_citation_rate_by_year(
    quiver: Quiver,
    metadata: dict,
    paper_id: str,
) -> pd.DataFrame:
    """
    Para un paper específico, retorna su conteo anual de citaciones recibidas.

    Returns
    -------
    DataFrame con columnas: year, n_citations
    """
    years = []
    for _, citing, cited in quiver.Q1:
        if cited == paper_id and citing in metadata:
            years.append(metadata[citing]["year"])

    if not years:
        return pd.DataFrame(columns=["year", "n_citations"])

    min_y = min(years)
    max_y = max(years)
    counts = {y: 0 for y in range(min_y, max_y + 1)}
    for y in years:
        counts[y] += 1

    return pd.DataFrame([
        {"year": y, "n_citations": c}
        for y, c in sorted(counts.items())
    ])


# ═══════════════════════════════════════════════════════════════════════
# Comparaciones AIQ vs realidad
# ═══════════════════════════════════════════════════════════════════════

def run_temporal_aiq_cohort(
    quiver: Quiver,
    metadata: dict,
    cohort_year: int,
    n_steps: int = 10,
    n_runs: int = 50,
    beta: float = 1.0,
    alpha: float = 1.0,
    recovery_prob: float = 0.3,
    P: Optional[Callable] = None,
    g_max: int = 3,
    seed: int = 42,
    parallel: bool = True,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ejecuta AIQ SIR para una cohorte de papers y rastrea la fracción
    que permanece en estado R (relevante) a lo largo del tiempo.

    Parameters
    ----------
    cohort_year : int
        Año de la cohorte. Papers de ese año empiezan como R.
    n_steps : int
        Pasos de simulación.
    n_runs : int
        Realizaciones Monte Carlo.
    parallel : bool
        Si True, paraleliza los runs Monte Carlo entre CPU cores.
    n_workers : int | None
        Número de procesos. Por defecto min(cpu_count, n_runs).

    Returns
    -------
    DataFrame con columnas: t, mean_fraction_R, std_fraction_R
    """
    if P is None:
        P = _default_P

    q0_set = set(quiver.Q0)
    cohort_papers = {
        pid for pid, m in metadata.items()
        if m["year"] == cohort_year and pid in q0_set
    }

    if not cohort_papers:
        raise ValueError(f"No hay papers de la cohorte {cohort_year} en el quiver")

    initial_config = {v: ("R" if v in cohort_papers else "S") for v in quiver.Q0}

    rng = np.random.default_rng(seed)
    run_seeds = rng.integers(0, 2**31, size=n_runs)

    fractions = np.zeros((n_runs, n_steps + 1))

    # Armar argumentos para cada run
    tasks = [
        (quiver, initial_config, cohort_papers, n_steps,
         beta, alpha, recovery_prob, P, g_max, int(run_seeds[r]))
        for r in range(n_runs)
    ]

    done = False
    if parallel and n_runs > 1:
        _nw = n_workers or min(os.cpu_count() or 4, n_runs)
        try:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=_nw, mp_context=ctx) as pool:
                results = list(pool.map(_mc_single_run, tasks))
            for r, fracs in enumerate(results):
                fractions[r, :len(fracs)] = fracs
            done = True
        except (RuntimeError, mp.context.BaseContext.AuthenticationError):
            # fork context not available (e.g. macOS, Windows); fall back
            pass

    if not done:
        for r, task in enumerate(tasks):
            fracs = _mc_single_run(task)
            fractions[r, :len(fracs)] = fracs

    rows = []
    for t in range(n_steps + 1):
        rows.append({
            "t": t,
            "mean_fraction_R": fractions[:, t].mean(),
            "std_fraction_R": fractions[:, t].std(),
        })
    return pd.DataFrame(rows)


def compare_impact_rate_vs_future_citations(
    quiver: Quiver,
    metadata: dict,
    snapshot_year: int,
    future_window: int = 3,
    beta: float = 1.0,
    alpha: float = 1.0,
    P: Optional[Callable] = None,
    g_max: int = 3,
) -> pd.DataFrame:
    """
    Correlaciona la tasa de impacto AIQ en snapshot_year con el número
    real de citaciones recibidas en los siguiente future_window años.

    Returns
    -------
    DataFrame con columnas: paper_id, impact_rate, future_citations, pub_year
    """
    from .impact import impact_rate_simple

    if P is None:
        P = _default_P

    # Papers publicados hasta snapshot_year
    valid_papers = {
        pid for pid, m in metadata.items()
        if m["year"] <= snapshot_year and pid in set(quiver.Q0)
    }

    if not valid_papers:
        raise ValueError(f"No hay papers hasta {snapshot_year}")

    # Configuración: papers publicados en snapshot_year son R, otros S
    config = {}
    for v in quiver.Q0:
        if v in valid_papers and metadata[v]["year"] == snapshot_year:
            config[v] = "R"
        else:
            config[v] = "S"

    # Calcular tasas de impacto para todos los S
    rates = {}
    for v in quiver.Q0:
        if v not in valid_papers:
            continue
        rate = impact_rate_simple(
            quiver, v, config, "R", beta, alpha, P, g_max
        )
        rates[v] = rate

    # Contar citaciones futuras reales
    future_citations = {pid: 0 for pid in valid_papers}
    for _, citing, cited in quiver.Q1:
        if cited in valid_papers and citing in metadata:
            y = metadata[citing]["year"]
            if snapshot_year < y <= snapshot_year + future_window:
                future_citations[cited] = future_citations.get(cited, 0) + 1

    rows = []
    for pid in valid_papers:
        if pid in rates:
            rows.append({
                "paper_id": pid,
                "impact_rate": rates[pid],
                "future_citations": future_citations.get(pid, 0),
                "pub_year": metadata[pid]["year"],
            })
    return pd.DataFrame(rows)


def validate_topological_traps(
    quiver: Quiver,
    metadata: dict,
    cohort_year: Optional[int] = None,
) -> dict:
    """
    Compara la vida útil de citación entre trampas topológicas (fuentes)
    y no-fuentes.

    Si cohort_year se especifica, restringe a papers de ese año para
    controlar por el efecto de la antigüedad.

    Returns
    -------
    dict con:
        'lifetime_df': DataFrame con lifetime por grupo
        'sources_median': float
        'non_sources_median': float
        'mann_whitney_U': float
        'mann_whitney_p': float
    """
    lt = compute_empirical_lifetime(quiver, metadata)

    if cohort_year is not None:
        lt = lt[lt["pub_year"] == cohort_year]

    sources = lt[lt["is_source"] == True]
    non_sources = lt[lt["is_source"] == False]

    result = {
        "lifetime_df": lt,
        "sources_median": sources["citation_lifetime_years"].median(),
        "non_sources_median": non_sources["citation_lifetime_years"].median(),
        "n_sources": len(sources),
        "n_non_sources": len(non_sources),
    }

    if len(sources) > 1 and len(non_sources) > 1:
        U, p = stats.mannwhitneyu(
            sources["citation_lifetime_years"].values,
            non_sources["citation_lifetime_years"].values,
            alternative="less",  # H1: fuentes tienen lifetime menor
        )
        result["mann_whitney_U"] = U
        result["mann_whitney_p"] = p
    else:
        result["mann_whitney_U"] = None
        result["mann_whitney_p"] = None

    return result


def validate_sfv_layer_contribution(
    quiver: Quiver,
    metadata: dict,
    sample_papers: int = 50,
    g_max: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Verifica si vecinos más cercanos (capas SFV con g pequeño) contribuyen
    más citaciones que vecinos lejanos.

    Para papers bien citados, calcula la fracción de papers en cada capa
    del SFV que realmente citan al paper objetivo.

    Returns
    -------
    DataFrame con columnas: g, mean_fraction_citing, std_fraction_citing, n_samples
    """
    # Seleccionar papers con suficientes citaciones
    in_degrees = {}
    for _, _, cited in quiver.Q1:
        in_degrees[cited] = in_degrees.get(cited, 0) + 1

    candidates = sorted(in_degrees, key=in_degrees.get, reverse=True)
    candidates = [p for p in candidates if in_degrees[p] >= 5]

    rng = np.random.default_rng(seed)
    if len(candidates) > sample_papers:
        idx = rng.choice(len(candidates), size=sample_papers, replace=False)
        candidates = [candidates[i] for i in idx]

    # Para cada candidato, analizar capas
    layer_fractions = {g: [] for g in range(1, g_max + 1)}

    # Construir set de citantes por paper
    citers_of = {}
    for _, citing, cited in quiver.Q1:
        citers_of.setdefault(cited, set()).add(citing)

    for target in candidates:
        fns = FundamentalNeighborhoodSystem(
            quiver, target, g_max=g_max, direction="in"
        )
        target_citers = citers_of.get(target, set())

        for g in range(1, g_max + 1):
            layer = fns.layer(g)
            if not layer:
                continue
            # Fracción de papers en capa g que citan al target
            n_citing = sum(1 for v in layer if v in target_citers)
            frac = n_citing / len(layer)
            layer_fractions[g].append(frac)

    rows = []
    for g in range(1, g_max + 1):
        vals = layer_fractions[g]
        if vals:
            rows.append({
                "g": g,
                "mean_fraction_citing": np.mean(vals),
                "std_fraction_citing": np.std(vals),
                "n_samples": len(vals),
            })
        else:
            rows.append({
                "g": g,
                "mean_fraction_citing": 0.0,
                "std_fraction_citing": 0.0,
                "n_samples": 0,
            })
    return pd.DataFrame(rows)


def validate_obsolescence_timing(
    quiver: Quiver,
    metadata: dict,
    cohort_years: Optional[list] = None,
    n_steps: int = 10,
    n_runs: int = 50,
    beta: float = 1.0,
    recovery_prob: float = 0.3,
    g_max: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compara el timing de obsolescencia predicho por AIQ con el empírico
    para varias cohortes.

    Para cada cohorte:
    - AIQ: paso t donde fraction_R < 0.5
    - Empírico: mediana del citation_lifetime

    Returns
    -------
    DataFrame con columnas: cohort_year, aiq_median_step, empirical_median_lifetime,
                            n_papers_cohort
    """
    if cohort_years is None:
        years = sorted({m["year"] for m in metadata.values()})
        # Excluir primeros y últimos 2 años (censoring)
        cohort_years = years[2:-2] if len(years) > 4 else years

    lt = compute_empirical_lifetime(quiver, metadata)

    rows = []
    for cy in cohort_years:
        # Empírico
        cohort_lt = lt[lt["pub_year"] == cy]
        if len(cohort_lt) < 5:
            continue
        emp_median = cohort_lt["citation_lifetime_years"].median()

        # AIQ
        try:
            aiq_df = run_temporal_aiq_cohort(
                quiver, metadata, cy,
                n_steps=n_steps, n_runs=n_runs,
                beta=beta, recovery_prob=recovery_prob,
                g_max=g_max, seed=seed,
            )
            # Encontrar paso donde fraction_R < 0.5
            below_half = aiq_df[aiq_df["mean_fraction_R"] < 0.5]
            aiq_step = below_half["t"].iloc[0] if len(below_half) > 0 else n_steps
        except (ValueError, Exception):
            aiq_step = None

        rows.append({
            "cohort_year": cy,
            "aiq_median_step": aiq_step,
            "empirical_median_lifetime": emp_median,
            "n_papers_cohort": len(cohort_lt),
        })

    return pd.DataFrame(rows)


def run_full_validation(
    quiver: Quiver,
    metadata: dict,
    max_age: int = 10,
    cohort_year: int = 1997,
    snapshot_year: int = 1998,
    n_runs: int = 50,
    beta: float = 1.0,
    recovery_prob: float = 0.3,
    g_max: int = 3,
    seed: int = 42,
) -> dict:
    """
    Ejecuta todas las validaciones y retorna un resumen.

    Returns
    -------
    dict con:
        'decay_empirical': DataFrame
        'decay_aiq': DataFrame
        'impact_correlation': DataFrame + stats
        'trap_validation': dict
        'sfv_validation': DataFrame
        'obsolescence_timing': DataFrame
        'summary': str
    """
    results = {}

    # 1. Curva de decaimiento empírica
    print("1/6 Calculando curva de decaimiento empírica...")
    results["decay_empirical"] = compute_empirical_decay_curve(
        quiver, metadata, max_age=max_age
    )

    # 2. Curva de decaimiento AIQ
    print("2/6 Ejecutando simulación AIQ para cohorte...")
    results["decay_aiq"] = run_temporal_aiq_cohort(
        quiver, metadata, cohort_year,
        n_steps=max_age, n_runs=n_runs,
        beta=beta, recovery_prob=recovery_prob,
        g_max=g_max, seed=seed,
    )

    # 3. Correlación tasa vs citaciones futuras
    print("3/6 Calculando correlación tasa-citaciones...")
    impact_df = compare_impact_rate_vs_future_citations(
        quiver, metadata, snapshot_year,
        beta=beta, g_max=g_max,
    )
    results["impact_correlation_df"] = impact_df
    # Calcular correlación
    valid = impact_df[impact_df["future_citations"] > 0]
    if len(valid) > 5:
        sp_r, sp_p = stats.spearmanr(valid["impact_rate"], valid["future_citations"])
        pe_r, pe_p = stats.pearsonr(valid["impact_rate"], valid["future_citations"])
    else:
        sp_r, sp_p, pe_r, pe_p = 0, 1, 0, 1
    results["impact_correlation_stats"] = {
        "spearman_r": sp_r, "spearman_p": sp_p,
        "pearson_r": pe_r, "pearson_p": pe_p,
        "n_papers": len(valid),
    }

    # 4. Trampas topológicas
    print("4/6 Validando trampas topológicas...")
    results["trap_validation"] = validate_topological_traps(
        quiver, metadata, cohort_year=cohort_year
    )

    # 5. Capas SFV
    print("5/6 Validando capas SFV...")
    results["sfv_validation"] = validate_sfv_layer_contribution(
        quiver, metadata, g_max=g_max, seed=seed,
    )

    # 6. Timing de obsolescencia
    print("6/6 Validando timing de obsolescencia...")
    results["obsolescence_timing"] = validate_obsolescence_timing(
        quiver, metadata,
        n_runs=n_runs, beta=beta, recovery_prob=recovery_prob,
        g_max=g_max, seed=seed,
    )

    # Resumen
    lines = [
        "=== Resumen de Validación Temporal AIQ ===",
        f"Papers: {quiver.n_vertices}, Aristas: {quiver.n_arrows}",
        f"",
        f"1. Correlación tasa-citaciones: Spearman ρ = {sp_r:.3f} (p = {sp_p:.4f})",
        f"2. Trampas topológicas: mediana fuentes = {results['trap_validation']['sources_median']:.1f} años, "
        f"no-fuentes = {results['trap_validation']['non_sources_median']:.1f} años",
    ]
    if results["trap_validation"]["mann_whitney_p"] is not None:
        lines.append(f"   Mann-Whitney p = {results['trap_validation']['mann_whitney_p']:.4f}")

    sfv = results["sfv_validation"]
    if len(sfv) > 0:
        fracs = sfv["mean_fraction_citing"].values
        is_decreasing = all(fracs[i] >= fracs[i+1] for i in range(len(fracs)-1))
        lines.append(f"3. Capas SFV: {'monotónica decreciente' if is_decreasing else 'NO monotónica'}")

    results["summary"] = "\n".join(lines)
    print(results["summary"])

    return results
