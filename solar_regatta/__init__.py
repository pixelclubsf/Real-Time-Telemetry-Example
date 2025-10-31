"""
Solar Regatta Telemetry Analysis Package

A Python package for analyzing and visualizing solar boat race telemetry data.
Includes web dashboard, data processing, and visualization tools.
"""

__version__ = "0.1.0"
__author__ = "Charlie Cullen"

from .core.analysis import (
    calculate_speeds,
    generate_sample_vesc_data,
    analyze_performance,
    plot_speed_vs_time,
    plot_with_coordinates,
    plot_all_metrics,
    dist
)

__all__ = [
    'calculate_speeds',
    'generate_sample_vesc_data',
    'analyze_performance',
    'plot_speed_vs_time',
    'plot_with_coordinates',
    'plot_all_metrics',
    'dist'
]
