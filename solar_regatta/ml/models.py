"""Machine-learning style utilities for Solar Regatta telemetry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from solar_regatta.core.analysis import calculate_speeds


@dataclass
class PerformanceModel:
    """Simple linear regression model for predicting boat speed."""

    coefficients: np.ndarray
    intercept: float
    feature_names: List[str]

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        feature_matrix = np.asarray(features, dtype=float)
        return feature_matrix @ self.coefficients + self.intercept


def _timestamps_to_seconds(timestamps: Sequence) -> np.ndarray:
    if not timestamps:
        return np.array([], dtype=float)
    first = timestamps[0]
    if hasattr(first, "timestamp"):
        base = first
        return np.array([(ts - base).total_seconds() for ts in timestamps], dtype=float)
    return np.asarray(timestamps, dtype=float)


def prepare_training_data(
    speeds: Sequence[float],
    battery_voltage: Sequence[float],
    motor_current: Sequence[float],
    timestamps: Sequence,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create a supervised dataset to predict the next speed sample."""

    speeds_arr = np.asarray(speeds, dtype=float)
    volts = np.asarray(battery_voltage, dtype=float)
    amps = np.asarray(motor_current, dtype=float)
    time_vals = _timestamps_to_seconds(timestamps)

    n_samples = min(len(speeds_arr) - 1, len(volts) - 1, len(amps) - 1)
    if n_samples <= 0:
        raise ValueError("Need at least two telemetry samples")

    # Use current measurements to predict the next speed sample.
    X = np.column_stack(
        [
            speeds_arr[:n_samples],
            volts[:n_samples],
            amps[:n_samples],
            np.diff(time_vals[: n_samples + 1]),
        ]
    )
    y = speeds_arr[1: n_samples + 1]

    feature_names = ["speed_t", "voltage_t", "current_t", "delta_time"]
    return X, y, feature_names


def train_speed_model(
    speeds: Sequence[float],
    battery_voltage: Sequence[float],
    motor_current: Sequence[float],
    timestamps: Sequence,
) -> PerformanceModel:
    """Fit a linear regression model that predicts future speed values."""

    X, y, feature_names = prepare_training_data(
        speeds, battery_voltage, motor_current, timestamps
    )

    # Append bias for intercept calculation.
    X_aug = np.column_stack([X, np.ones(len(X))])
    solution, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    coefficients = solution[:-1]
    intercept = solution[-1]

    return PerformanceModel(coefficients=coefficients, intercept=intercept, feature_names=feature_names)


def evaluate_model(model: PerformanceModel, X: np.ndarray, y: np.ndarray) -> dict:
    """Generate basic regression metrics."""

    predictions = model.predict(X)
    residuals = y - predictions
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return {"mse": mse, "mae": mae, "r2": r2}


def forecast_speed_curve(model: PerformanceModel, feature_matrix: np.ndarray) -> np.ndarray:
    """Predict a sequence of speeds from prepared feature matrix."""

    return model.predict(feature_matrix)


def train_on_raw_gps(
    gps_points: Sequence[str],
    timestamps: Sequence,
    battery_voltage: Sequence[float],
    motor_current: Sequence[float],
) -> Tuple[PerformanceModel, np.ndarray, np.ndarray]:
    """Convenience helper that derives speeds from GPS inputs then trains."""

    speeds = calculate_speeds(gps_points, timestamps)
    X, y, _ = prepare_training_data(speeds, battery_voltage, motor_current, timestamps)
    model = train_speed_model(speeds, battery_voltage, motor_current, timestamps)
    return model, X, y
