"""Advanced feature engineering for telemetry data."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class FeatureEngineer:
    """Advanced feature engineering for time-series telemetry data.

    Generates rolling statistics, lag features, rate of change,
    and physics-based derived features.
    """

    def __init__(
        self,
        rolling_windows: List[int] = [3, 5, 10],
        lag_features: int = 3,
        include_derivatives: bool = True,
        include_physics: bool = True,
    ):
        """Initialize feature engineer.

        Args:
            rolling_windows: Window sizes for rolling statistics
            lag_features: Number of lag features to create
            include_derivatives: Include rate of change features
            include_physics: Include physics-based features
        """
        self.rolling_windows = rolling_windows
        self.lag_features = lag_features
        self.include_derivatives = include_derivatives
        self.include_physics = include_physics
        self.feature_names_: Optional[List[str]] = None

    def create_rolling_features(
        self,
        values: np.ndarray,
        name: str,
        windows: List[int]
    ) -> Tuple[np.ndarray, List[str]]:
        """Create rolling statistics features.

        Args:
            values: Input time series
            name: Feature name prefix
            windows: Window sizes

        Returns:
            Feature matrix and feature names
        """
        features = []
        names = []

        for window in windows:
            # Rolling mean
            rolling_mean = np.convolve(
                values, np.ones(window)/window, mode='valid'
            )
            # Pad to match original length
            rolling_mean = np.pad(
                rolling_mean,
                (window-1, 0),
                mode='edge'
            )
            features.append(rolling_mean)
            names.append(f"{name}_rolling_mean_{window}")

            # Rolling std
            rolling_std = np.array([
                np.std(values[max(0, i-window+1):i+1])
                for i in range(len(values))
            ])
            features.append(rolling_std)
            names.append(f"{name}_rolling_std_{window}")

            # Rolling max
            rolling_max = np.array([
                np.max(values[max(0, i-window+1):i+1])
                for i in range(len(values))
            ])
            features.append(rolling_max)
            names.append(f"{name}_rolling_max_{window}")

            # Rolling min
            rolling_min = np.array([
                np.min(values[max(0, i-window+1):i+1])
                for i in range(len(values))
            ])
            features.append(rolling_min)
            names.append(f"{name}_rolling_min_{window}")

        return np.column_stack(features), names

    def create_lag_features(
        self,
        values: np.ndarray,
        name: str,
        n_lags: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Create lag features.

        Args:
            values: Input time series
            name: Feature name prefix
            n_lags: Number of lags

        Returns:
            Feature matrix and feature names
        """
        features = []
        names = []

        for lag in range(1, n_lags + 1):
            lagged = np.roll(values, lag)
            lagged[:lag] = lagged[lag]  # Fill initial values
            features.append(lagged)
            names.append(f"{name}_lag_{lag}")

        return np.column_stack(features), names

    def create_derivative_features(
        self,
        values: np.ndarray,
        name: str,
        dt: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Create rate of change features.

        Args:
            values: Input time series
            name: Feature name prefix
            dt: Time differences (if None, assumes uniform)

        Returns:
            Feature matrix and feature names
        """
        features = []
        names = []

        # First derivative
        if dt is not None:
            derivative = np.gradient(values, dt)
        else:
            derivative = np.gradient(values)
        features.append(derivative)
        names.append(f"{name}_derivative")

        # Second derivative (acceleration)
        if dt is not None:
            second_derivative = np.gradient(derivative, dt)
        else:
            second_derivative = np.gradient(derivative)
        features.append(second_derivative)
        names.append(f"{name}_acceleration")

        return np.column_stack(features), names

    def create_physics_features(
        self,
        speeds: np.ndarray,
        voltages: np.ndarray,
        currents: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """Create physics-based derived features.

        Args:
            speeds: Speed values
            voltages: Voltage values
            currents: Current values

        Returns:
            Feature matrix and feature names
        """
        features = []
        names = []

        # Power consumption
        power = voltages * currents
        features.append(power)
        names.append("power_consumption")

        # Motor efficiency (speed per amp)
        efficiency = np.where(currents > 0.1, speeds / currents, 0)
        features.append(efficiency)
        names.append("motor_efficiency")

        # Energy efficiency (speed per watt)
        energy_eff = np.where(power > 0.1, speeds / power, 0)
        features.append(energy_eff)
        names.append("energy_efficiency")

        # Voltage-speed ratio
        voltage_speed = np.where(speeds > 0.1, voltages / speeds, 0)
        features.append(voltage_speed)
        names.append("voltage_speed_ratio")

        # Current-speed ratio
        current_speed = np.where(speeds > 0.1, currents / speeds, 0)
        features.append(current_speed)
        names.append("current_speed_ratio")

        # Cumulative energy
        cumulative_energy = np.cumsum(power)
        features.append(cumulative_energy)
        names.append("cumulative_energy")

        return np.column_stack(features), names

    def fit_transform(
        self,
        speeds: np.ndarray,
        voltages: np.ndarray,
        currents: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Create all features from telemetry data.

        Args:
            speeds: Speed measurements
            voltages: Voltage measurements
            currents: Current measurements
            timestamps: Timestamps (optional)

        Returns:
            Feature matrix and feature names
        """
        all_features = []
        all_names = []

        # Original features
        all_features.extend([speeds, voltages, currents])
        all_names.extend(['speed', 'voltage', 'current'])

        # Time delta if timestamps provided
        if timestamps is not None:
            dt = np.diff(timestamps, prepend=timestamps[0])
            all_features.append(dt)
            all_names.append('delta_time')
        else:
            dt = None

        # Rolling features
        for values, name in [(speeds, 'speed'), (voltages, 'voltage'), (currents, 'current')]:
            rolling_feats, rolling_names = self.create_rolling_features(
                values, name, self.rolling_windows
            )
            for i in range(rolling_feats.shape[1]):
                all_features.append(rolling_feats[:, i])
                all_names.append(rolling_names[i])

        # Lag features
        if self.lag_features > 0:
            for values, name in [(speeds, 'speed'), (voltages, 'voltage'), (currents, 'current')]:
                lag_feats, lag_names = self.create_lag_features(
                    values, name, self.lag_features
                )
                for i in range(lag_feats.shape[1]):
                    all_features.append(lag_feats[:, i])
                    all_names.append(lag_names[i])

        # Derivative features
        if self.include_derivatives:
            for values, name in [(speeds, 'speed'), (voltages, 'voltage'), (currents, 'current')]:
                deriv_feats, deriv_names = self.create_derivative_features(
                    values, name, dt
                )
                for i in range(deriv_feats.shape[1]):
                    all_features.append(deriv_feats[:, i])
                    all_names.append(deriv_names[i])

        # Physics features
        if self.include_physics:
            phys_feats, phys_names = self.create_physics_features(
                speeds, voltages, currents
            )
            for i in range(phys_feats.shape[1]):
                all_features.append(phys_feats[:, i])
                all_names.append(phys_names[i])

        # Stack all features
        X = np.column_stack(all_features)
        self.feature_names_ = all_names

        return X, all_names

    def get_feature_names(self) -> List[str]:
        """Get feature names after fitting.

        Returns:
            List of feature names
        """
        if self.feature_names_ is None:
            raise ValueError("Must call fit_transform first")
        return self.feature_names_


def create_polynomial_features(
    X: np.ndarray,
    degree: int = 2,
    interaction_only: bool = False,
) -> np.ndarray:
    """Create polynomial and interaction features.

    Args:
        X: Input features
        degree: Polynomial degree
        interaction_only: Only create interaction terms

    Returns:
        Augmented feature matrix
    """
    from itertools import combinations_with_replacement

    n_samples, n_features = X.shape
    features = [X]

    if interaction_only:
        # Only pairwise interactions
        for i, j in combinations_with_replacement(range(n_features), 2):
            if i != j:
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))
    else:
        # Full polynomial
        for d in range(2, degree + 1):
            for indices in combinations_with_replacement(range(n_features), d):
                term = np.prod(X[:, indices], axis=1).reshape(-1, 1)
                features.append(term)

    return np.hstack(features)
