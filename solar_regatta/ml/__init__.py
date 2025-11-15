"""ML utilities for Solar Regatta."""

from .models import (
    PerformanceModel,
    evaluate_model,
    forecast_speed_curve,
    prepare_training_data,
    train_on_raw_gps,
    train_speed_model,
)

__all__ = [
    "PerformanceModel",
    "evaluate_model",
    "forecast_speed_curve",
    "prepare_training_data",
    "train_on_raw_gps",
    "train_speed_model",
]
