"""
Autómata de Impacto sobre Quivers (AIQ).

Referencia: ACT.tex, Definición 3.1 (AIQ), Definición 2.9–2.11 (reglas),
Ejemplo 3.1 (simulación SIS), Ejemplo 4.1 (citación).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .quiver import Quiver
from .impact import (
    _LayerCache,
    impact_rate_simple,
    impact_rate_enriched,
    impact_rate_signed,
    effective_transition_probability,
)


class AIQ:
    """
    Autómata de Impacto sobre un Quiver.

    A = (Q, Σ, {A_k}, φ)

    donde Q es el quiver, Σ el conjunto de estados, {A_k} el SFV,
    y φ la regla de evolución.

    Parameters
    ----------
    quiver : Quiver
        El quiver subyacente Q.
    states : list
        Conjunto de estados Σ (p.ej. ['S', 'I'] o ['S', 'I', 'R']).
    evolution_rule : str
        'SI', 'SIS' o 'SIR'.
    beta : float
        Tasa de infección/activación.
    alpha : float
        Factor de normalización (típicamente 1.0).
    P : callable
        Función de peso P(g) → float. P(0)=1, decreciente.
    g_max : int or None
        Radio máximo del SFV. None → usa diámetro.
    impact_mode : str
        'simple' (Def. 2.9), 'enriched' (Def. 2.10), 'signed' (Def. 2.11).
    quotient : QuotientAlgebra or None
        Cociente kQ/I para tasa enriquecida algebraica.
    recovery_prob : float
        Probabilidad de recuperación γ para SIS/SIR (Def. 2.10).
    fixed_states : dict or None
        {vértice: estado} para vértices con estado forzado.
    """

    def __init__(
        self,
        quiver: Quiver,
        states: list,
        evolution_rule: str = "SIS",
        beta: float = 1.0,
        alpha: float = 1.0,
        P: Optional[Callable[[int], float]] = None,
        g_max: Optional[int] = None,
        impact_mode: str = "simple",
        quotient=None,
        recovery_prob: float = 0.5,
        fixed_states: Optional[dict] = None,
    ):
        self.quiver = quiver
        self.states = list(states)
        self.evolution_rule = evolution_rule.upper()
        self.beta = beta
        self.alpha = alpha
        self.P = P if P is not None else (lambda g: 1.0 / (g + 1))
        self.g_max = g_max if g_max is not None else quiver.diameter()
        self.impact_mode = impact_mode
        self.quotient = quotient
        self.recovery_prob = recovery_prob
        self.fixed_states = fixed_states or {}

        # Estado actual y órbita
        self._config: dict = {}
        self._orbit: list[dict] = []

        # Pre-compute layer cache for simple mode (the hot path)
        self._layer_cache: Optional[_LayerCache] = None
        if impact_mode == "simple":
            self._layer_cache = _LayerCache(quiver, self.g_max, direction="in")

        # Validar
        if self.evolution_rule not in ("SI", "SIS", "SIR"):
            raise ValueError(f"Regla desconocida: {self.evolution_rule}")

    # ── Configuración ─────────────────────────────────────────────────

    def set_initial_config(self, config: dict):
        """
        Establece la configuración inicial π^0.

        Parameters
        ----------
        config : dict
            {vértice: estado} para cada vértice del quiver.
        """
        for v in self.quiver.Q0:
            if v not in config:
                raise ValueError(f"Falta estado para vértice '{v}'")
            if config[v] not in self.states:
                raise ValueError(
                    f"Estado '{config[v]}' no está en Σ = {self.states}"
                )
        self._config = dict(config)
        # Aplicar estados fijos
        for v, s in self.fixed_states.items():
            self._config[v] = s
        self._orbit = [dict(self._config)]

    @property
    def config(self) -> dict:
        """Configuración actual."""
        return dict(self._config)

    @property
    def orbit(self) -> list[dict]:
        """Secuencia completa de configuraciones (órbita)."""
        return list(self._orbit)

    @property
    def time(self) -> int:
        """Paso temporal actual."""
        return len(self._orbit) - 1

    # ── Tasa de impacto ───────────────────────────────────────────────

    def _compute_rate(self, cell, target_state) -> float:
        """Calcula la tasa de impacto según el modo seleccionado."""
        if self.impact_mode == "simple":
            return impact_rate_simple(
                self.quiver, cell, self._config, target_state,
                self.beta, self.alpha, self.P, self.g_max,
                _layer_cache=self._layer_cache,
            )
        elif self.impact_mode == "enriched":
            return impact_rate_enriched(
                self.quiver, cell, self._config, target_state,
                self.beta, self.alpha, self.P, self.g_max,
                quotient=self.quotient,
            )
        elif self.impact_mode == "signed":
            return impact_rate_signed(
                self.quiver, cell, self._config, target_state,
                self.beta, self.alpha, self.P, self.g_max,
            )
        else:
            raise ValueError(f"Modo de impacto desconocido: {self.impact_mode}")

    # ── Reglas de evolución (vectorizadas por batch random) ───────────

    def _apply_rule(self, rng: np.random.Generator) -> dict:
        """
        Aplica la regla de evolución a todos los vértices sincrónicamente.
        Genera todos los números aleatorios en batch para eficiencia.
        """
        verts = self.quiver.Q0
        n = len(verts)
        new_config = dict(self._config)

        # Batch random numbers
        rands = rng.random(n)

        state_S = self.states[0]
        state_I = self.states[1]
        has_R = len(self.states) > 2
        state_R = self.states[2] if has_R else None
        rule = self.evolution_rule

        for i, v in enumerate(verts):
            if v in self.fixed_states:
                new_config[v] = self.fixed_states[v]
                continue

            current = self._config[v]

            if current == state_S:
                rate = self._compute_rate(v, state_I)
                p = effective_transition_probability(rate)
                if rands[i] < p:
                    new_config[v] = state_I

            elif current == state_I:
                if rule == "SIS":
                    if rands[i] < self.recovery_prob:
                        new_config[v] = state_S
                elif rule == "SIR" and has_R:
                    if rands[i] < self.recovery_prob:
                        new_config[v] = state_R
                # SI: I stays I (no action needed)

        return new_config

    # ── Evolución ─────────────────────────────────────────────────────

    def step(self, seed: Optional[int] = None) -> dict:
        """
        Ejecuta un paso de evolución sincrónica.

        Parameters
        ----------
        seed : int or None
            Semilla para reproducibilidad.

        Returns
        -------
        dict
            Nueva configuración.
        """
        rng = np.random.default_rng(seed)
        new_config = self._apply_rule(rng)
        self._config = new_config
        self._orbit.append(dict(new_config))
        return dict(new_config)

    def run(self, n_steps: int, seed: Optional[int] = None) -> list[dict]:
        """
        Ejecuta n_steps pasos de evolución.

        Usa semillas deterministas derivadas de la semilla base para
        reproducibilidad completa.

        Parameters
        ----------
        n_steps : int
            Número de pasos.
        seed : int or None
            Semilla base.

        Returns
        -------
        list[dict]
            Órbita completa (incluyendo configuración inicial).
        """
        base_rng = np.random.default_rng(seed)
        step_seeds = base_rng.integers(0, 2**31, size=n_steps)
        for i in range(n_steps):
            self.step(seed=int(step_seeds[i]))
        return list(self._orbit)

    # ── Estadísticas ──────────────────────────────────────────────────

    def state_counts(self, config: Optional[dict] = None) -> dict:
        """Cuenta de cada estado en la configuración."""
        cfg = config or self._config
        counts = {s: 0 for s in self.states}
        for v in cfg.values():
            counts[v] = counts.get(v, 0) + 1
        return counts

    def orbit_table(self) -> pd.DataFrame:
        """
        Tabla de la órbita como DataFrame de pandas.
        Filas = pasos temporales, columnas = vértices.
        """
        verts = self.quiver.Q0
        data = []
        for t, cfg in enumerate(self._orbit):
            row = {"t": t}
            for v in verts:
                row[str(v)] = cfg[v]
            data.append(row)
        return pd.DataFrame(data).set_index("t")

    def orbit_counts_table(self) -> pd.DataFrame:
        """
        Tabla de conteo por estados a lo largo de la órbita.
        Filas = pasos temporales, columnas = estados.
        """
        data = []
        for t, cfg in enumerate(self._orbit):
            row = {"t": t}
            counts = self.state_counts(cfg)
            row.update(counts)
            data.append(row)
        return pd.DataFrame(data).set_index("t")

    def run_statistics(
        self,
        initial_config: dict,
        n_steps: int,
        n_runs: int = 100,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Estadísticas Monte Carlo: ejecuta n_runs simulaciones independientes.

        Returns
        -------
        pd.DataFrame
            Columnas: t, estado, mean, std, min, max.
        """
        base_rng = np.random.default_rng(seed)
        run_seeds = base_rng.integers(0, 2**31, size=n_runs)

        n_states = len(self.states)
        counts_all = np.zeros((n_runs, n_steps + 1, n_states), dtype=np.int32)

        for r in range(n_runs):
            self.set_initial_config(initial_config)
            self.run(n_steps, seed=int(run_seeds[r]))
            for t, cfg in enumerate(self._orbit):
                cnts = self.state_counts(cfg)
                for s_idx, s in enumerate(self.states):
                    counts_all[r, t, s_idx] = cnts[s]

        rows = []
        for t in range(n_steps + 1):
            for s_idx, s in enumerate(self.states):
                vals = counts_all[:, t, s_idx]
                rows.append({
                    "t": t,
                    "state": s,
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "min": vals.min(),
                    "max": vals.max(),
                })
        return pd.DataFrame(rows)

    # ── Utilidades topológicas ────────────────────────────────────────

    def topologically_trapped_vertices(self) -> list:
        """
        Vértices que son fuentes (in-degree 0): trampas topológicas.
        En un modelo SI/SIS, una vez que se desactivan no pueden
        ser reactivadas por impacto. (Ejemplo 3.1)
        """
        return [v for v in self.quiver.Q0 if self.quiver.is_source(v)]

    def impact_rates_snapshot(self, target_state=None) -> dict:
        """
        Diccionario {vértice: tasa_de_impacto} en la configuración actual.
        Útil para diagnóstico y visualización.
        """
        if target_state is None:
            target_state = self.states[1]
        rates = {}
        for v in self.quiver.Q0:
            rates[v] = self._compute_rate(v, target_state)
        return rates

    # ── Representación ────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AIQ(rule={self.evolution_rule}, "
            f"|Q_0|={self.quiver.n_vertices}, "
            f"Σ={self.states}, "
            f"mode={self.impact_mode}, "
            f"t={self.time})"
        )
