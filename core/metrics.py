"""Metrics calculation module for portfolio analysis."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioMetrics:
    """Container for portfolio metrics."""
    name: str
    annual_return: float
    annual_volatility: float
    sharpe: float
    cagr: float
    return_12m: float
    beta_spy: float
    beta_qqq: float


def calculate_metrics(
    returns: pd.Series,
    rf: float = 0.02
) -> Tuple[float, float, float, float, float]:
    """
    Calculate key portfolio metrics.

    Args:
        returns: Daily returns series
        rf: Annual risk-free rate

    Returns:
        Tuple of (annual_return, annual_vol, sharpe, cagr, return_12m)
    """
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - rf) / annual_vol if annual_vol != 0 else 0

    cum = (1 + returns).cumprod()
    cagr = (cum.iloc[-1]) ** (252 / len(returns)) - 1

    if len(returns) > 252:
        return_12m = (cum.iloc[-1] / cum.iloc[-252]) - 1
    else:
        return_12m = np.nan

    return annual_return, annual_vol, sharpe, cagr, return_12m


def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate portfolio beta relative to a benchmark.

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns

    Returns:
        Beta coefficient
    """
    if len(portfolio_returns) != len(benchmark_returns):
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns.iloc[-min_len:]
        benchmark_returns = benchmark_returns.iloc[-min_len:]

    cov = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns)
    return cov / var if var != 0 else 0


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    days: int = 1
) -> float:
    """
    Calculate Value at Risk using parametric method.

    Args:
        returns: Daily returns series
        confidence: Confidence level (default 95%)
        days: Holding period in days

    Returns:
        VaR as a positive percentage
    """
    mean = returns.mean()
    std = returns.std()
    var_1d = -(mean + norm.ppf(1 - confidence) * std)
    var_nd = var_1d * np.sqrt(days)
    return var_nd


def calculate_cagr(cumulative_returns: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        cumulative_returns: Cumulative returns series (not percentage)

    Returns:
        CAGR as decimal
    """
    total_periods = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25
    if total_periods <= 0:
        return 0
    total_return = cumulative_returns.iloc[-1] + 1
    return total_return ** (1 / total_periods) - 1


def get_portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
    assets: List[str]
) -> pd.Series:
    """
    Calculate weighted portfolio returns.

    Args:
        returns: DataFrame with asset returns
        weights: Portfolio weights
        assets: List of asset tickers

    Returns:
        Series of portfolio daily returns
    """
    return returns[assets].dot(weights)


def build_metrics_table(
    portfolio_returns: Dict[str, pd.Series],
    benchmark_returns: pd.DataFrame,
    rf: float = 0.02
) -> pd.DataFrame:
    """
    Build a comprehensive metrics comparison table.

    Args:
        portfolio_returns: Dict mapping portfolio names to returns series
        benchmark_returns: DataFrame with benchmark returns
        rf: Risk-free rate

    Returns:
        DataFrame with all metrics
    """
    rows = []

    for name, returns in portfolio_returns.items():
        ann_ret, ann_vol, sharpe, cagr, ret_12m = calculate_metrics(returns, rf)

        beta_spy = calculate_beta(returns, benchmark_returns['SPY']) if 'SPY' in benchmark_returns else np.nan
        beta_qqq = calculate_beta(returns, benchmark_returns['QQQ']) if 'QQQ' in benchmark_returns else np.nan

        rows.append({
            'Portfolio': name,
            'Retorno Anual (%)': ann_ret * 100,
            'Volatilidad (%)': ann_vol * 100,
            'Sharpe': sharpe,
            'CAGR (%)': cagr * 100,
            'Retorno 12m (%)': ret_12m * 100 if not np.isnan(ret_12m) else np.nan,
            'Beta SPY': beta_spy,
            'Beta QQQ': beta_qqq
        })

    for bench in benchmark_returns.columns:
        ann_ret, ann_vol, sharpe, cagr, ret_12m = calculate_metrics(benchmark_returns[bench], rf)
        rows.append({
            'Portfolio': bench,
            'Retorno Anual (%)': ann_ret * 100,
            'Volatilidad (%)': ann_vol * 100,
            'Sharpe': sharpe,
            'CAGR (%)': cagr * 100,
            'Retorno 12m (%)': ret_12m * 100 if not np.isnan(ret_12m) else np.nan,
            'Beta SPY': 1.0 if bench == 'SPY' else np.nan,
            'Beta QQQ': 1.0 if bench == 'QQQ' else np.nan
        })

    return pd.DataFrame(rows)


def build_var_table(
    portfolio_returns: Dict[str, pd.Series],
    spy_returns: pd.Series,
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Build Value at Risk table.

    Args:
        portfolio_returns: Dict mapping portfolio names to returns series
        spy_returns: SPY benchmark returns
        confidence: Confidence level

    Returns:
        DataFrame with VaR metrics
    """
    rows = []

    for name, returns in portfolio_returns.items():
        var_1d = calculate_var(returns, confidence, 1) * 100
        var_10d = calculate_var(returns, confidence, 10) * 100
        rows.append({
            'Portfolio': name,
            'VaR 1 dia (%)': var_1d,
            'VaR 10 dias (%)': var_10d
        })

    var_spy_1d = calculate_var(spy_returns, confidence, 1) * 100
    var_spy_10d = calculate_var(spy_returns, confidence, 10) * 100
    rows.append({
        'Portfolio': 'SPY',
        'VaR 1 dia (%)': var_spy_1d,
        'VaR 10 dias (%)': var_spy_10d
    })

    return pd.DataFrame(rows)
