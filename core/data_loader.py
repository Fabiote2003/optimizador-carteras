"""Data loading and validation module."""
import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Tuple, List


@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers: List[str], period: str) -> pd.DataFrame:
    """
    Download historical price data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        period: Time period (e.g., '5y' for 5 years)

    Returns:
        DataFrame with closing prices
    """
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    data = data.dropna(axis=1, how='all')

    return data


def validate_tickers(tickers: List[str], data: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Validate which tickers have data available.

    Args:
        tickers: List of requested tickers
        data: Downloaded data DataFrame

    Returns:
        Tuple of (valid_tickers, invalid_tickers)
    """
    valid = [t for t in tickers if t in data.columns]
    invalid = [t for t in tickers if t not in data.columns]
    return valid, invalid


def prepare_data(tickers: List[str], years: int, benchmarks: List[str] = None) -> dict:
    """
    Download and prepare all data needed for optimization.

    Args:
        tickers: List of asset tickers
        years: Number of years of history
        benchmarks: List of benchmark tickers (default: ['SPY', 'QQQ'])

    Returns:
        Dictionary with data, returns, and metadata
    """
    if benchmarks is None:
        benchmarks = ['SPY', 'QQQ']

    all_tickers = tickers.copy()
    for b in benchmarks:
        if b not in all_tickers:
            all_tickers.append(b)

    period = f"{years}y"
    data = download_data(all_tickers, period)

    valid_assets, invalid_assets = validate_tickers(tickers, data)
    valid_benchmarks = [b for b in benchmarks if b in data.columns]

    if len(valid_assets) == 0:
        raise ValueError("No hay datos disponibles para los tickers ingresados.")

    data = data.dropna()

    returns = data.pct_change().dropna()

    mean_returns = returns[valid_assets].mean() * 252
    cov_matrix = returns[valid_assets].cov() * 252

    return {
        'data': data,
        'returns': returns,
        'assets': valid_assets,
        'invalid_assets': invalid_assets,
        'benchmarks': valid_benchmarks,
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix
    }
