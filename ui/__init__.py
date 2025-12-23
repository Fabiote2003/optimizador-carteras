"""UI modules for Streamlit app."""
from .charts import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_correlation_matrix,
    plot_cumulative_returns,
    plot_cagr_comparison,
    plot_var_histograms
)
from .components import display_metrics_table, display_var_table, display_stress_table
