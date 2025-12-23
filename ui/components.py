"""UI components for Streamlit app."""
import pandas as pd
import streamlit as st
from typing import List, Tuple


def display_metrics_table(df: pd.DataFrame) -> None:
    """
    Display styled metrics comparison table.

    Args:
        df: DataFrame with portfolio metrics
    """
    styled = df.style.format({
        'Retorno Anual (%)': '{:.2f}',
        'Volatilidad (%)': '{:.2f}',
        'Sharpe': '{:.2f}',
        'CAGR (%)': '{:.2f}',
        'Retorno 12m (%)': '{:.2f}',
        'Beta SPY': '{:.2f}',
        'Beta QQQ': '{:.2f}'
    }).background_gradient(
        subset=['Retorno Anual (%)'], cmap='Greens'
    ).background_gradient(
        subset=['Volatilidad (%)'], cmap='Reds_r'
    ).background_gradient(
        subset=['Sharpe'], cmap='YlGn'
    ).background_gradient(
        subset=['CAGR (%)'], cmap='Greens'
    ).set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    })

    st.dataframe(styled, width="stretch", hide_index=True)


def display_var_table(df: pd.DataFrame) -> None:
    """
    Display Value at Risk table.

    Args:
        df: DataFrame with VaR metrics
    """
    styled = df.style.format({
        'VaR 1 dia (%)': '{:.2f}',
        'VaR 10 dias (%)': '{:.2f}'
    }).background_gradient(
        subset=['VaR 1 dia (%)', 'VaR 10 dias (%)'], cmap='OrRd'
    ).set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    })

    st.dataframe(styled, width="stretch", hide_index=True)


def display_stress_table(df: pd.DataFrame, title: str = "") -> None:
    """
    Display stress test results table.

    Args:
        df: DataFrame with stress test results
        title: Optional title for the table
    """
    if title:
        st.subheader(title)

    styled = df.style.format('{:.2f}%').background_gradient(
        axis=None, cmap='OrRd', vmin=df.min().min(), vmax=0
    ).set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    })

    st.dataframe(styled, width="stretch")


def display_weights_table(weights_df: pd.DataFrame) -> None:
    """
    Display portfolio weights table.

    Args:
        weights_df: DataFrame with asset weights
    """
    st.dataframe(
        weights_df,
        width="stretch",
        hide_index=True
    )


def display_high_correlation_warning(high_corr: List[Tuple[str, str, float]]) -> None:
    """
    Display warning for high correlations.

    Args:
        high_corr: List of (asset1, asset2, correlation) tuples
    """
    if high_corr:
        st.error("**ATENCION: DIVERSIFICACION INSUFICIENTE**")
        st.warning("Se detectaron activos con correlacion > 0.80:")
        for a1, a2, corr in high_corr:
            st.write(f"- **{a1}** - **{a2}**: correlacion {corr:.2f}")
    else:
        st.success("**DIVERSIFICACION CORRECTA** - Ninguna correlacion > 0.80")


def display_portfolio_summary(
    name: str,
    returns: float,
    volatility: float,
    sharpe: float
) -> None:
    """
    Display portfolio summary metrics in columns.

    Args:
        name: Portfolio name
        returns: Annual returns
        volatility: Annual volatility
        sharpe: Sharpe ratio
    """
    st.subheader(name)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Retorno Anual", f"{returns*100:.2f}%")
    with col2:
        st.metric("Volatilidad", f"{volatility*100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")


def display_var_interpretation() -> None:
    """Display interpretation help for VaR charts."""
    st.info("""
    **Como interpretar cada grafico:**
    - Cada histograma muestra la frecuencia de retornos diarios del portfolio.
    - La linea discontinua marca el **Value at Risk (VaR)** al 95%.
    - Hay solo un 5% de chances de que la perdida diaria sea peor que ese valor.
    - Ejemplo: si el VaR es 2.5%, con 95% de confianza NO se espera perder mas de 2.5% en un solo dia.
    """)
