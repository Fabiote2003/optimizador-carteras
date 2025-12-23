"""Chart generation module for portfolio visualization."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import streamlit as st


plt.style.use('seaborn-v0_8-whitegrid')


def plot_efficient_frontier(
    random_portfolios: Dict,
    frontier: Dict,
    portfolios: List[Dict],
    rf: float = 0.02
) -> plt.Figure:
    """
    Plot the efficient frontier with random portfolios and optimal points.

    Args:
        random_portfolios: Dict with returns, volatility, sharpe arrays
        frontier: Dict with returns and volatility arrays
        portfolios: List of portfolio dicts with name, returns, volatility
        rf: Risk-free rate

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    sc = ax.scatter(
        random_portfolios['volatility'],
        random_portfolios['returns'],
        c=random_portfolios['sharpe'],
        cmap='viridis',
        alpha=0.5,
        s=10,
        label='Portfolios Aleatorios'
    )
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

    valid_mask = ~np.isnan(frontier['volatility'])
    ax.plot(
        frontier['volatility'][valid_mask],
        frontier['returns'][valid_mask],
        'b-',
        linewidth=3,
        label='Frontera Eficiente'
    )

    markers = ['*', 'o', 'X', 's', 'D']
    colors = ['gold', 'red', 'green', 'purple', 'orange']
    sizes = [300, 150, 150, 150, 150]

    for i, port in enumerate(portfolios):
        ax.scatter(
            port['volatility'],
            port['returns'],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            s=sizes[i % len(sizes)],
            label=port['name'],
            edgecolors='black',
            linewidth=1,
            zorder=5
        )

    if portfolios:
        max_sharpe_port = max(portfolios, key=lambda x: x.get('sharpe', 0))
        sharpe = max_sharpe_port.get('sharpe', 0)
        vol_max = np.nanmax(frontier['volatility']) * 1.2
        vol_cml = np.linspace(0, vol_max, 100)
        cml_line = rf + sharpe * vol_cml
        ax.plot(vol_cml, cml_line, 'r--', linewidth=2, label='CML')

    ax.set_title('Espacio de Portfolios (Markowitz)', fontsize=16, weight='bold')
    ax.set_xlabel('Volatilidad Anual', fontsize=12)
    ax.set_ylabel('Retorno Anual', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_portfolio_weights(
    weights: np.ndarray,
    assets: List[str],
    title: str,
    color: str = '#1976D2'
) -> plt.Figure:
    """
    Plot horizontal bar chart of portfolio weights.

    Args:
        weights: Array of portfolio weights
        assets: List of asset names
        title: Chart title
        color: Primary color for the chart

    Returns:
        Matplotlib figure
    """
    df = pd.DataFrame({'Activo': assets, 'Peso (%)': np.array(weights) * 100})
    df = df[df['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=True)

    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No hay activos con peso significativo',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))

    bars = ax.barh(df['Activo'], df['Peso (%)'], color=colors, edgecolor='black', height=0.7)

    ax.set_xlim(0, max(100, df['Peso (%)'].max() + 10))
    ax.set_xlabel('Peso (%)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold', pad=15)

    for idx, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                f'{width:.1f}%', va='center', fontsize=11, fontweight='bold')

    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.set_ylabel('')

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    returns: pd.DataFrame,
    assets: List[str]
) -> Tuple[plt.Figure, List[Tuple[str, str, float]]]:
    """
    Plot correlation matrix heatmap.

    Args:
        returns: DataFrame with asset returns
        assets: List of assets to include

    Returns:
        Tuple of (figure, list of high correlations)
    """
    corr_matrix = returns[assets].corr()

    cmap_custom = LinearSegmentedColormap.from_list(
        'CelesteRojoInvert',
        ['lightblue', 'red'],
        N=256
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 10},
        cmap=cmap_custom,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor='gray',
        square=True,
        cbar_kws={'shrink': 0.8, 'pad': 0.02, 'label': 'Correlacion'},
        ax=ax
    )

    ax.set_title('Matriz de Correlacion entre Activos', fontsize=14, weight='bold')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.80:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    plt.tight_layout()
    return fig, high_corr


def plot_cumulative_returns(
    portfolio_returns: Dict[str, pd.Series],
    benchmark_returns: pd.DataFrame
) -> plt.Figure:
    """
    Plot cumulative returns comparison.

    Args:
        portfolio_returns: Dict mapping names to returns series
        benchmark_returns: DataFrame with benchmark returns

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1976D2', '#C62828', '#388E3C', '#7B1FA2']
    for i, (name, returns) in enumerate(portfolio_returns.items()):
        cum = (1 + returns).cumprod() - 1
        ax.plot(cum.index, cum * 100, label=name, linewidth=2,
                color=colors[i % len(colors)])

    for col in benchmark_returns.columns:
        cum = (1 + benchmark_returns[col]).cumprod() - 1
        ax.plot(cum.index, cum * 100, label=col, linestyle='--', linewidth=1.5)

    ax.set_title('Rendimientos Acumulados: Portfolios vs Benchmarks', fontsize=14, weight='bold')
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Crecimiento Acumulado (%)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cagr_comparison(
    cagr_data: Dict[str, Dict[str, float]]
) -> plt.Figure:
    """
    Plot CAGR comparison bar chart.

    Args:
        cagr_data: Dict with 'assets', 'portfolios', 'benchmarks' subdicts

    Returns:
        Matplotlib figure
    """
    data = []
    for name, cagr in cagr_data.get('assets', {}).items():
        data.append({'Nombre': name, 'CAGR (%)': cagr * 100, 'Tipo': 'Activo'})
    for name, cagr in cagr_data.get('portfolios', {}).items():
        data.append({'Nombre': name, 'CAGR (%)': cagr * 100, 'Tipo': 'Portfolio'})
    for name, cagr in cagr_data.get('benchmarks', {}).items():
        data.append({'Nombre': name, 'CAGR (%)': cagr * 100, 'Tipo': 'Benchmark'})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='Nombre', y='CAGR (%)', hue='Tipo', dodge=False, ax=ax)

    ax.set_title('Comparativa CAGR Anual', fontsize=14, weight='bold')
    ax.set_ylabel('CAGR Anual (%)', fontsize=12)
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Tipo', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_var_histograms(
    portfolio_returns: Dict[str, pd.Series],
    var_1d: Dict[str, float]
) -> plt.Figure:
    """
    Plot histograms of returns with VaR lines.

    Args:
        portfolio_returns: Dict mapping names to returns series
        var_1d: Dict mapping names to 1-day VaR percentages

    Returns:
        Matplotlib figure
    """
    n_plots = len(portfolio_returns)
    cols = 2
    rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    colors = ['#1976D2', '#C62828', '#388E3C', '#f0ad4e']

    for i, (name, returns) in enumerate(portfolio_returns.items()):
        if i >= len(axes):
            break

        ax = axes[i]
        data = returns * 100

        sns.histplot(data, bins=60, color=colors[i % len(colors)],
                     kde=True, stat='density', alpha=0.35, ax=ax)

        var_val = var_1d.get(name, 0)
        ax.axvline(-var_val, color=colors[i % len(colors)],
                   linestyle='--', linewidth=2)
        ax.text(-var_val, ax.get_ylim()[1] * 0.80,
                f'VaR\n{var_val:.2f}%',
                color=colors[i % len(colors)],
                fontsize=10, ha='right', va='top', fontweight='bold')

        ax.set_title(f'Retornos Diarios & VaR: {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Retorno Diario (%)')
        ax.set_ylabel('Densidad')
        ax.grid(alpha=0.25)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def plot_weights_table(
    weights: np.ndarray,
    assets: List[str]
) -> pd.DataFrame:
    """
    Create a DataFrame of portfolio weights for display.

    Args:
        weights: Array of weights
        assets: List of asset names

    Returns:
        DataFrame formatted for display
    """
    df = pd.DataFrame({
        'Activo': assets,
        'Peso (%)': [f'{w*100:.2f}' for w in weights]
    })
    df = df[df['Peso (%)'].astype(float) > 0.01]
    return df.reset_index(drop=True)
