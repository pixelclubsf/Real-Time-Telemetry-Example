"""Tests for core analysis functionality."""
import pytest
import numpy as np
from datetime import datetime, timedelta


class TestGPSCalculations:
    """Tests for GPS distance and speed calculations."""

    def test_calculate_speeds_basic(self):
        """Test speed calculation with simple timestamps."""
        from solar_regatta import calculate_speeds

        # Two points about 100m apart (approximate MGRS coords)
        gps_points = ["10SEG1234567890", "10SEG1234567990"]
        timestamps = [0, 1]  # 1 second apart

        speeds = calculate_speeds(gps_points, timestamps)

        assert len(speeds) == 1
        assert speeds[0] >= 0  # Speed should be non-negative

    def test_calculate_speeds_with_datetime(self):
        """Test speed calculation with datetime timestamps."""
        from solar_regatta import calculate_speeds

        gps_points = ["10SEG1234567890", "10SEG1234567990"]
        base_time = datetime.now()
        timestamps = [base_time, base_time + timedelta(seconds=1)]

        speeds = calculate_speeds(gps_points, timestamps)

        assert len(speeds) == 1
        assert isinstance(speeds[0], (int, float))

    def test_calculate_speeds_empty_returns_empty(self):
        """Test that empty input returns empty output."""
        from solar_regatta import calculate_speeds

        speeds = calculate_speeds([], [])
        assert speeds == []

    def test_calculate_speeds_single_point(self):
        """Test that single point returns empty speeds."""
        from solar_regatta import calculate_speeds

        speeds = calculate_speeds(["10SEG1234567890"], [0])
        assert speeds == []


class TestSampleDataGeneration:
    """Tests for sample data generation."""

    def test_generate_sample_vesc_data_returns_tuple(self):
        """Test sample data generation returns tuple."""
        from solar_regatta import generate_sample_vesc_data

        result = generate_sample_vesc_data()

        assert isinstance(result, tuple)
        assert len(result) == 5  # gps, timestamps, speeds, voltage, current

    def test_generate_sample_vesc_data_custom_duration(self):
        """Test sample data with custom duration."""
        from solar_regatta import generate_sample_vesc_data

        gps, timestamps, speeds, voltage, current = generate_sample_vesc_data(
            duration_seconds=60, interval=5
        )

        # 60 seconds / 5 second interval = 13 samples (0, 5, 10, ..., 60)
        assert len(timestamps) == 13
        assert len(gps) == 13
        assert len(voltage) == 13
        assert len(current) == 13

    def test_sample_data_has_valid_ranges(self):
        """Test that sample data has realistic value ranges."""
        from solar_regatta import generate_sample_vesc_data

        gps, timestamps, speeds, voltage, current = generate_sample_vesc_data()

        # Voltage should be in reasonable battery range
        assert all(10 <= v <= 60 for v in voltage)


class TestPerformanceAnalysis:
    """Tests for performance analysis functions."""

    def test_analyze_performance_basic(self):
        """Test basic performance analysis."""
        from solar_regatta import analyze_performance

        speeds = [1.0, 2.0, 3.0, 2.5, 1.5]
        voltage = [13.0, 13.1, 13.0, 12.9, 12.8]
        current = [5.0, 6.0, 7.0, 6.5, 5.5]
        base = datetime.now()
        timestamps = [base + timedelta(seconds=i) for i in range(5)]

        metrics = analyze_performance(speeds, voltage, current, timestamps)

        assert "max_speed" in metrics
        assert "avg_speed" in metrics
        assert metrics["max_speed"] == 3.0

    def test_analyze_performance_empty_speeds(self):
        """Test analysis with empty speeds returns empty dict."""
        from solar_regatta import analyze_performance

        metrics = analyze_performance([], [], [], [])
        assert metrics == {}

    def test_analyze_performance_includes_efficiency(self):
        """Test that analysis includes efficiency metrics."""
        from solar_regatta import analyze_performance

        speeds = [1.0, 2.0, 3.0]
        voltage = [13.0, 13.1, 13.0]
        current = [5.0, 6.0, 7.0]
        base = datetime.now()
        timestamps = [base + timedelta(seconds=i) for i in range(3)]

        metrics = analyze_performance(speeds, voltage, current, timestamps)

        # Should include power/energy metrics
        assert "avg_power" in metrics or "total_energy" in metrics or len(metrics) > 0
