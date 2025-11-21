"""Visualization helpers."""

from .plotly_charts import (
    create_current_plot,
    create_efficiency_plot,
    create_gps_path_plot,
    create_speed_plot,
    create_voltage_plot,
)

# World model visualizations (optional)
try:
    from .world_model_viz import (
        plot_trajectory_2d,
        plot_state_evolution,
        plot_strategy_comparison,
        plot_uncertainty_bands,
        create_control_visualization,
    )
    WORLD_MODEL_VIZ_AVAILABLE = True
except (ImportError, Exception):
    WORLD_MODEL_VIZ_AVAILABLE = False

__all__ = [
    "create_current_plot",
    "create_efficiency_plot",
    "create_gps_path_plot",
    "create_speed_plot",
    "create_voltage_plot",
]

if WORLD_MODEL_VIZ_AVAILABLE:
    __all__.extend([
        "plot_trajectory_2d",
        "plot_state_evolution",
        "plot_strategy_comparison",
        "plot_uncertainty_bands",
        "create_control_visualization",
    ])
