"""
Visualización de quivers, evolución de AIQs, y heatmaps.

Usa matplotlib + networkx para dibujo.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import pandas as pd

from .quiver import Quiver


# ═══════════════════════════════════════════════════════════════════════
# Colores por defecto
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_STATE_COLORS = {
    "S": "#4CAF50",   # Verde
    "I": "#F44336",   # Rojo
    "R": "#2196F3",   # Azul
    "A": "#F44336",   # Activo (rojo)
    "E": "#FF9800",   # Expresado (naranja)
    "O": "#9E9E9E",   # Obsoleto (gris)
}


# ═══════════════════════════════════════════════════════════════════════
# Dibujo de quivers
# ═══════════════════════════════════════════════════════════════════════

def draw_quiver(
    quiver: Quiver,
    config: Optional[dict] = None,
    state_colors: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    pos: Optional[dict] = None,
    node_size: int = 800,
    font_size: int = 10,
    arrow_style: str = "-|>",
    title: Optional[str] = None,
    show_arrow_labels: bool = False,
    show_weights: bool = False,
    figsize: tuple = (8, 6),
) -> tuple:
    """
    Dibuja un quiver con colores según la configuración.

    Parameters
    ----------
    quiver : Quiver
    config : dict or None
        {vértice: estado} para colorear nodos.
    state_colors : dict
        {estado: color_hex}.
    pos : dict or None
        {vértice: (x, y)}. Si None, usa spring_layout.
    show_arrow_labels : bool
        Mostrar nombres de flechas.
    show_weights : bool
        Mostrar pesos de flechas.

    Returns
    -------
    fig, ax
    """
    import networkx as nx

    colors = {**DEFAULT_STATE_COLORS, **(state_colors or {})}
    G = quiver.to_networkx()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=2)

    # Colores de nodos
    node_colors = []
    for v in G.nodes():
        if config and v in config:
            state = config[v]
            node_colors.append(colors.get(state, "#9E9E9E"))
        else:
            node_colors.append("#E0E0E0")

    # Dibujar nodos
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors,
        node_size=node_size, edgecolors="black", linewidths=1.5,
    )

    # Etiquetas de nodos
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=font_size, font_weight="bold",
    )

    # Dibujar aristas con connectionstyle para flechas múltiples
    edge_counts = {}
    for _, src, tgt in quiver.Q1:
        key = (src, tgt)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    drawn = {}
    for name, src, tgt in quiver.Q1:
        key = (src, tgt)
        n_edges = edge_counts[key]
        idx = drawn.get(key, 0)
        drawn[key] = idx + 1

        if n_edges == 1:
            rad = 0.0
        else:
            rad = 0.2 * (idx - (n_edges - 1) / 2)

        w = quiver.arrow_weight(name)
        edge_color = "#333333" if w >= 0 else "#E53935"
        style = "solid" if w >= 0 else "dashed"

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(src, tgt)],
            connectionstyle=f"arc3,rad={rad}",
            edge_color=edge_color,
            style=style,
            arrows=True,
            arrowstyle=arrow_style,
            arrowsize=15,
            min_source_margin=20,
            min_target_margin=20,
            width=1.5,
        )

        if show_arrow_labels:
            # Posición intermedia
            x = (pos[src][0] + pos[tgt][0]) / 2 + rad * 0.3
            y = (pos[src][1] + pos[tgt][1]) / 2 + rad * 0.3
            label = name
            if show_weights and w != 1:
                label += f" ({w:+d})"
            ax.text(x, y, label, fontsize=7, ha="center",
                    color=edge_color, style="italic")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# Evolución temporal
# ═══════════════════════════════════════════════════════════════════════

def draw_quiver_evolution(
    aiq,
    time_steps: Optional[list[int]] = None,
    pos: Optional[dict] = None,
    state_colors: Optional[dict] = None,
    ncols: int = 4,
    figsize_per_subplot: tuple = (3.5, 3),
) -> plt.Figure:
    """
    Grid de snapshots del quiver en varios instantes.

    Parameters
    ----------
    aiq : AIQ
        Autómata con órbita calculada.
    time_steps : list[int] or None
        Instantes a mostrar. None → todos.
    """
    orbit = aiq.orbit
    if time_steps is None:
        time_steps = list(range(len(orbit)))

    n = len(time_steps)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    import networkx as nx
    if pos is None:
        G = aiq.quiver.to_networkx()
        pos = nx.spring_layout(G, seed=42, k=2)

    for idx, t in enumerate(time_steps):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        config = orbit[t] if t < len(orbit) else orbit[-1]
        draw_quiver(
            aiq.quiver, config=config,
            state_colors=state_colors, ax=ax, pos=pos,
            node_size=500, font_size=8,
            title=f"t = {t}",
        )

    # Ocultar ejes vacíos
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    return fig


def plot_evolution_counts(
    aiq,
    state_colors: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 4),
    title: str = "Evolución de estados",
) -> tuple:
    """
    Curvas de conteo de cada estado vs tiempo.
    """
    colors = {**DEFAULT_STATE_COLORS, **(state_colors or {})}
    df = aiq.orbit_counts_table()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    for state in aiq.states:
        if state in df.columns:
            ax.plot(
                df.index, df[state],
                marker="o", label=state,
                color=colors.get(state, None),
                linewidth=2, markersize=5,
            )

    ax.set_xlabel("Tiempo t", fontsize=11)
    ax.set_ylabel("Conteo", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(aiq.orbit)))

    return fig, ax


def plot_evolution_heatmap(
    aiq,
    state_encoding: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    title: str = "Heatmap de evolución",
    cmap: str = "RdYlGn_r",
) -> tuple:
    """
    Heatmap vértices × tiempo con estados codificados numéricamente.
    """
    if state_encoding is None:
        state_encoding = {s: i for i, s in enumerate(aiq.states)}

    orbit = aiq.orbit
    verts = aiq.quiver.Q0
    T = len(orbit)
    n = len(verts)

    matrix = np.zeros((n, T), dtype=np.float64)
    for t, cfg in enumerate(orbit):
        for v_idx, v in enumerate(verts):
            matrix[v_idx, t] = state_encoding.get(cfg[v], 0)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   interpolation="nearest")
    ax.set_xlabel("Tiempo t", fontsize=11)
    ax.set_ylabel("Vértice", fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(v) for v in verts], fontsize=8)
    ax.set_xticks(range(T))
    ax.set_title(title, fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, ticks=sorted(state_encoding.values()))
    cbar.ax.set_yticklabels(
        [s for s, _ in sorted(state_encoding.items(), key=lambda x: x[1])]
    )

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# Matrices de impacto
# ═══════════════════════════════════════════════════════════════════════

def plot_impact_matrix(
    quiver: Quiver,
    k: int = 1,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 5),
    cmap: str = "YlOrRd",
    title: Optional[str] = None,
) -> tuple:
    """
    Heatmap de A^k (walks de longitud k).
    """
    Ak = quiver.adjacency_power(k)
    labels = [str(v) for v in quiver.Q0]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(Ak, cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    fig.colorbar(im, ax=ax)

    # Anotar valores
    for i in range(Ak.shape[0]):
        for j in range(Ak.shape[1]):
            val = Ak[i, j]
            if val != 0:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, color="white" if val > Ak.max() / 2 else "black")

    if title is None:
        title = f"$A^{{{k}}}$ — walks de longitud {k}"
    ax.set_title(title, fontsize=13, fontweight="bold")

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# Estadísticas Monte Carlo
# ═══════════════════════════════════════════════════════════════════════

def plot_statistics(
    stats_df: pd.DataFrame,
    state_colors: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    title: str = "Estadísticas Monte Carlo",
    show_band: bool = True,
) -> tuple:
    """
    Media ± desviación estándar de conteos de estados.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Salida de AIQ.run_statistics().
    """
    colors = {**DEFAULT_STATE_COLORS, **(state_colors or {})}

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    for state in stats_df["state"].unique():
        df_s = stats_df[stats_df["state"] == state].sort_values("t")
        c = colors.get(state, None)
        ax.plot(df_s["t"], df_s["mean"], label=state, color=c,
                linewidth=2)
        if show_band:
            ax.fill_between(
                df_s["t"],
                df_s["mean"] - df_s["std"],
                df_s["mean"] + df_s["std"],
                alpha=0.2, color=c,
            )

    ax.set_xlabel("Tiempo t", fontsize=11)
    ax.set_ylabel("Conteo (media ± σ)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# Tabla de órbita estilizada
# ═══════════════════════════════════════════════════════════════════════

def orbit_table_styled(
    aiq,
    state_colors: Optional[dict] = None,
) -> pd.io.formats.style.Styler:
    """
    DataFrame de la órbita con celdas coloreadas por estado.
    Para mostrar en Jupyter.
    """
    colors = {**DEFAULT_STATE_COLORS, **(state_colors or {})}
    df = aiq.orbit_table()

    def color_cell(val):
        bg = colors.get(val, "#FFFFFF")
        # Determinar contraste
        try:
            r, g, b = mcolors.to_rgb(bg)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if lum < 0.5 else "black"
        except Exception:
            text_color = "black"
        return f"background-color: {bg}; color: {text_color}; font-weight: bold"

    return df.style.map(color_cell)


def plot_sfv_layers(
    quiver: Quiver,
    cell,
    fns,
    pos: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 6),
    title: Optional[str] = None,
    cmap_name: str = "coolwarm",
) -> tuple:
    """
    Visualiza las capas del SFV coloreadas por distancia.

    Parameters
    ----------
    fns : FundamentalNeighborhoodSystem
    """
    import networkx as nx

    G = quiver.to_networkx()
    if pos is None:
        pos = nx.spring_layout(G, seed=42, k=2)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    cmap = plt.cm.get_cmap(cmap_name, fns.g_max + 1)

    node_colors = []
    for v in quiver.Q0:
        found = False
        for g in range(fns.g_max + 1):
            if v in fns.layer(g):
                node_colors.append(cmap(g / max(fns.g_max, 1)))
                found = True
                break
        if not found:
            node_colors.append("#CCCCCC")

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors,
        node_size=800, edgecolors="black", linewidths=1.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#666666",
        arrows=True, arrowstyle="-|>", arrowsize=15,
        min_source_margin=20, min_target_margin=20,
    )

    if title is None:
        title = f"SFV de {cell}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()

    # Leyenda
    for g in range(fns.g_max + 1):
        layer = fns.layer(g)
        if layer:
            ax.plot([], [], "o", color=cmap(g / max(fns.g_max, 1)),
                    markersize=10, label=f"N_{g}: {layer}")
    ax.legend(loc="best", fontsize=8)

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
# Visualización de validación temporal
# ═══════════════════════════════════════════════════════════════════════

def plot_citation_age_distribution(
    ages_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    title: str = "Distribución de edades de citación",
    bins: int = 30,
) -> tuple:
    """Histograma de edades de citación (años entre citante y citado)."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    ages = ages_df["citation_age_years"]
    ax.hist(ages[ages >= 0], bins=bins, color="#5C6BC0",
            edgecolor="white", alpha=0.8, density=True)
    ax.set_xlabel("Edad de citación (años)", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(ages[ages >= 0].median(), color="#E53935",
               linestyle="--", linewidth=2,
               label=f"Mediana = {ages[ages >= 0].median():.1f} años")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_decay_comparison(
    empirical_df: pd.DataFrame,
    aiq_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 6),
    title: str = "Curva de decaimiento: Empírica vs AIQ",
    step_to_year: float = 1.0,
) -> tuple:
    """
    Superpone la curva de supervivencia empírica con la predicción AIQ.

    Parameters
    ----------
    empirical_df : DataFrame
        Salida de compute_empirical_decay_curve().
    aiq_df : DataFrame
        Salida de run_temporal_aiq_cohort().
    step_to_year : float
        Factor de conversión: 1 paso AIQ = step_to_year años.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Empírica
    ax.plot(empirical_df["age"], empirical_df["fraction_cited"],
            "o-", color="#1565C0", linewidth=2, markersize=6,
            label="Empírica (fracción citada)")

    # AIQ
    t_years = aiq_df["t"] * step_to_year
    ax.plot(t_years, aiq_df["mean_fraction_R"],
            "s--", color="#C62828", linewidth=2, markersize=6,
            label="AIQ SIR (fracción R)")

    if "std_fraction_R" in aiq_df.columns:
        ax.fill_between(
            t_years,
            aiq_df["mean_fraction_R"] - aiq_df["std_fraction_R"],
            aiq_df["mean_fraction_R"] + aiq_df["std_fraction_R"],
            alpha=0.15, color="#C62828",
        )

    ax.set_xlabel("Edad / Tiempo (años)", fontsize=11)
    ax.set_ylabel("Fracción activa", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    return fig, ax


def plot_impact_rate_scatter(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 7),
    title: str = "Tasa de impacto AIQ vs citaciones futuras",
    log_scale: bool = True,
) -> tuple:
    """
    Scatter plot: tasa de impacto AIQ vs citaciones futuras reales.

    Parameters
    ----------
    df : DataFrame
        Salida de compare_impact_rate_vs_future_citations().
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    valid = df[df["future_citations"] > 0].copy()

    sc = ax.scatter(
        valid["impact_rate"], valid["future_citations"],
        c=valid["pub_year"], cmap="viridis", alpha=0.5, s=20,
        edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label="Año de publicación")

    if log_scale and len(valid) > 0:
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("log")

    # Correlación
    if len(valid) > 5:
        from scipy import stats as sp_stats
        rho, p = sp_stats.spearmanr(valid["impact_rate"], valid["future_citations"])
        ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\np = {p:.2e}\nn = {len(valid)}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Tasa de impacto AIQ", fontsize=11)
    ax.set_ylabel("Citaciones futuras (reales)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_trap_comparison(
    trap_result: dict,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    title: str = "Trampas topológicas: lifetime de citación",
) -> tuple:
    """
    Box plot comparando lifetime de fuentes vs no-fuentes.

    Parameters
    ----------
    trap_result : dict
        Salida de validate_topological_traps().
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    lt = trap_result["lifetime_df"]
    sources = lt[lt["is_source"] == True]["citation_lifetime_years"]
    non_sources = lt[lt["is_source"] == False]["citation_lifetime_years"]

    bp = ax.boxplot(
        [sources.values, non_sources.values],
        labels=[f"Fuentes\n(n={len(sources)})",
                f"No-fuentes\n(n={len(non_sources)})"],
        patch_artist=True,
        widths=0.5,
    )

    colors_bp = ["#EF5350", "#42A5F5"]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Anotación
    p_val = trap_result.get("mann_whitney_p")
    if p_val is not None:
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text(0.5, 0.95,
                f"Mann-Whitney p = {p_val:.4f} ({sig})",
                transform=ax.transAxes, ha="center", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_ylabel("Lifetime de citación (años)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    return fig, ax


def plot_sfv_layer_decay(
    sfv_df: pd.DataFrame,
    P=None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
    title: str = "Contribución de citación por capa SFV",
) -> tuple:
    """
    Barras: fracción de citación por capa g del SFV.

    Parameters
    ----------
    sfv_df : DataFrame
        Salida de validate_sfv_layer_contribution().
    P : callable or None
        Función de peso teórica P(g) para superponer.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.bar(sfv_df["g"], sfv_df["mean_fraction_citing"],
           yerr=sfv_df["std_fraction_citing"],
           color="#5C6BC0", alpha=0.7, capsize=5,
           label="Empírica")

    if P is not None:
        g_vals = sfv_df["g"].values
        p_vals = [P(g) for g in g_vals]
        # Normalizar para comparación visual
        max_emp = sfv_df["mean_fraction_citing"].max()
        max_p = max(p_vals) if max(p_vals) > 0 else 1
        p_scaled = [p * max_emp / max_p for p in p_vals]
        ax.plot(g_vals, p_scaled, "ro--", linewidth=2, markersize=8,
                label="P(g) teórica (escalada)")

    ax.set_xlabel("Capa SFV (distancia g)", fontsize=11)
    ax.set_ylabel("Fracción que cita al target", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(sfv_df["g"].values)

    return fig, ax


def plot_obsolescence_timing(
    timing_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 6),
    title: str = "Timing de obsolescencia: AIQ vs Empírico",
) -> tuple:
    """
    Scatter: timing AIQ vs empírico por cohorte.

    Parameters
    ----------
    timing_df : DataFrame
        Salida de validate_obsolescence_timing().
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    valid = timing_df.dropna(subset=["aiq_median_step"])

    ax.scatter(valid["empirical_median_lifetime"],
               valid["aiq_median_step"],
               s=100, c="#1565C0", edgecolors="black", linewidths=1,
               zorder=5)

    # Etiquetas de cohorte
    for _, row in valid.iterrows():
        ax.annotate(str(int(row["cohort_year"])),
                     (row["empirical_median_lifetime"], row["aiq_median_step"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)

    # Línea de referencia
    if len(valid) > 0:
        max_val = max(valid["empirical_median_lifetime"].max(),
                      valid["aiq_median_step"].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3,
                label="Acuerdo perfecto")

    ax.set_xlabel("Mediana empírica (años)", fontsize=11)
    ax.set_ylabel("Mediana AIQ (pasos)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_validation_dashboard(
    results: dict,
    P=None,
    figsize: tuple = (18, 12),
) -> plt.Figure:
    """
    Panel 3x2 con todas las validaciones.

    Parameters
    ----------
    results : dict
        Salida de run_full_validation().
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    plot_decay_comparison(
        results["decay_empirical"], results["decay_aiq"],
        ax=axes[0, 0], title="Decaimiento: Empírica vs AIQ",
    )

    plot_impact_rate_scatter(
        results["impact_correlation_df"],
        ax=axes[0, 1], title="Tasa AIQ vs citaciones",
    )

    plot_trap_comparison(
        results["trap_validation"],
        ax=axes[0, 2], title="Trampas topológicas",
    )

    plot_sfv_layer_decay(
        results["sfv_validation"], P=P,
        ax=axes[1, 0], title="Capas SFV",
    )

    if len(results["obsolescence_timing"]) > 0:
        plot_obsolescence_timing(
            results["obsolescence_timing"],
            ax=axes[1, 1], title="Timing de obsolescencia",
        )
    else:
        axes[1, 1].text(0.5, 0.5, "Sin datos", ha="center", va="center")

    axes[1, 2].axis("off")
    summary = results.get("summary", "Sin resumen")
    axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                    fontsize=9, verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Dashboard de Validación Temporal AIQ",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
