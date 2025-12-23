"""Stress testing module for portfolio analysis."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .metrics import calculate_beta


HISTORICAL_SCENARIOS = [
    {
        "name": "Crisis 2008 (Lehman)",
        "shock_spy": -0.09,
        "description": "SPY -9% en 1 dia (Lehman 2008)"
    },
    {
        "name": "COVID-19 Crash",
        "shock_spy": -0.12,
        "description": "SPY -12% en 1 dia (COVID-19 2020)"
    }
]

HYPOTHETICAL_SHOCKS = [-0.05, -0.10, -0.20]


def calculate_portfolio_betas(
    portfolio_returns: Dict[str, pd.Series],
    spy_returns: pd.Series
) -> Dict[str, float]:
    """
    Calculate beta for each portfolio.

    Args:
        portfolio_returns: Dict mapping portfolio names to returns series
        spy_returns: SPY benchmark returns

    Returns:
        Dict mapping portfolio names to beta values
    """
    betas = {}
    for name, returns in portfolio_returns.items():
        betas[name] = calculate_beta(returns, spy_returns)
    betas['SPY'] = 1.0
    return betas


def run_stress_test(
    betas: Dict[str, float],
    shocks: List[float] = None
) -> pd.DataFrame:
    """
    Run stress test with hypothetical market shocks.

    Args:
        betas: Dict mapping portfolio names to beta values
        shocks: List of shock percentages (default: [-5%, -10%, -20%])

    Returns:
        DataFrame with stress test results
    """
    if shocks is None:
        shocks = HYPOTHETICAL_SHOCKS

    results = {}
    for name, beta in betas.items():
        impacts = [beta * shock * 100 for shock in shocks]
        results[name] = impacts

    columns = [f"SPY {shock*100:.0f}%" for shock in shocks]
    df = pd.DataFrame(results, index=columns).T
    df.index.name = 'Portfolio'

    return df


def run_historical_stress_test(
    betas: Dict[str, float],
    scenarios: List[Dict] = None
) -> pd.DataFrame:
    """
    Run stress test with historical crisis scenarios.

    Args:
        betas: Dict mapping portfolio names to beta values
        scenarios: List of scenario dictionaries (default: HISTORICAL_SCENARIOS)

    Returns:
        DataFrame with historical stress test results
    """
    if scenarios is None:
        scenarios = HISTORICAL_SCENARIOS

    results = {}
    for name, beta in betas.items():
        impacts = [beta * scenario['shock_spy'] * 100 for scenario in scenarios]
        results[name] = impacts

    columns = [scenario['name'] for scenario in scenarios]
    df = pd.DataFrame(results, index=columns).T
    df.index.name = 'Portfolio'

    return df


def get_stress_interpretation() -> str:
    """Return interpretation text for stress test results."""
    return """
    **Como interpretar:**
    - Cada celda muestra la caida estimada si el SPY baja ese porcentaje en un dia.
    - Se usa la **beta** calculada para estimar la sensibilidad al mercado.
    - Ejemplo: Si la celda dice **-7.8%** y el SPY baja -10%, ese portfolio caeria aproximadamente -7.8% ese dia.
    """
