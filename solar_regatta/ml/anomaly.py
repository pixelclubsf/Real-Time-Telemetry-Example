"""Anomaly detection for telemetry data."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class SimpleAnomalyDetector:
    """Statistical anomaly detection using Z-score and IQR methods."""

    def __init__(
        self,
        method: str = 'zscore',
        threshold: float = 3.0,
    ):
        """Initialize anomaly detector.

        Args:
            method: Detection method ('zscore' or 'iqr')
            threshold: Threshold for anomaly detection
        """
        self.method = method
        self.threshold = threshold
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self.q1_: Optional[float] = None
        self.q3_: Optional[float] = None

    def fit(self, X: np.ndarray):
        """Fit the detector on normal data.

        Args:
            X: Training data
        """
        if self.method == 'zscore':
            self.mean_ = np.mean(X)
            self.std_ = np.std(X)
        elif self.method == 'iqr':
            self.q1_ = np.percentile(X, 25)
            self.q3_ = np.percentile(X, 75)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies.

        Args:
            X: Data to check

        Returns:
            Boolean array (True = anomaly)
        """
        if self.method == 'zscore':
            if self.mean_ is None or self.std_ is None:
                raise ValueError("Detector not fitted")
            z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-8))
            return z_scores > self.threshold

        elif self.method == 'iqr':
            if self.q1_ is None or self.q3_ is None:
                raise ValueError("Detector not fitted")
            iqr = self.q3_ - self.q1_
            lower_bound = self.q1_ - self.threshold * iqr
            upper_bound = self.q3_ + self.threshold * iqr
            return (X < lower_bound) | (X > upper_bound)

        return np.zeros(len(X), dtype=bool)


def detect_voltage_anomalies(
    voltages: np.ndarray,
    low_threshold: float = 10.5,
    high_threshold: float = 14.0,
    rate_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect battery voltage anomalies.

    Args:
        voltages: Voltage measurements
        low_threshold: Low voltage threshold
        high_threshold: High voltage threshold
        rate_threshold: Maximum voltage change rate (V/sample)

    Returns:
        Tuple of (low_voltage, high_voltage, rapid_change) boolean arrays
    """
    low_voltage = voltages < low_threshold
    high_voltage = voltages > high_threshold

    voltage_changes = np.abs(np.diff(voltages, prepend=voltages[0]))
    rapid_change = voltage_changes > rate_threshold

    return low_voltage, high_voltage, rapid_change


def detect_current_spikes(
    currents: np.ndarray,
    spike_threshold: float = 3.0,
    window_size: int = 5,
) -> np.ndarray:
    """Detect motor current spikes.

    Args:
        currents: Current measurements
        spike_threshold: Z-score threshold for spikes
        window_size: Window for rolling statistics

    Returns:
        Boolean array of spike locations
    """
    # Rolling mean and std
    rolling_mean = np.convolve(
        currents,
        np.ones(window_size)/window_size,
        mode='same'
    )
    rolling_std = np.array([
        np.std(currents[max(0, i-window_size):min(len(currents), i+window_size)])
        for i in range(len(currents))
    ])

    # Detect spikes
    z_scores = np.abs((currents - rolling_mean) / (rolling_std + 1e-8))
    return z_scores > spike_threshold


def detect_gps_anomalies(
    speeds: np.ndarray,
    max_speed: float = 15.0,
    max_acceleration: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect GPS/speed anomalies.

    Args:
        speeds: Speed measurements
        max_speed: Maximum realistic speed (m/s)
        max_acceleration: Maximum acceleration (m/s²)

    Returns:
        Tuple of (impossible_speed, impossible_accel) boolean arrays
    """
    impossible_speed = speeds > max_speed

    acceleration = np.abs(np.diff(speeds, prepend=speeds[0]))
    impossible_accel = acceleration > max_acceleration

    return impossible_speed, impossible_accel
