"""Tests for feature engineering module."""
import pytest
import numpy as np

from solar_regatta.ml import FeatureEngineer


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_feature_engineer_creation(self):
        """Test FeatureEngineer can be created with defaults."""
        fe = FeatureEngineer()

        assert fe.rolling_windows == [3, 5, 10]
        assert fe.lag_features == 3
        assert fe.include_derivatives is True

    def test_feature_engineer_custom_params(self):
        """Test FeatureEngineer with custom parameters."""
        fe = FeatureEngineer(
            rolling_windows=[5, 10],
            lag_features=5,
            include_derivatives=False
        )

        assert fe.rolling_windows == [5, 10]
        assert fe.lag_features == 5
        assert fe.include_derivatives is False

    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        fe = FeatureEngineer(rolling_windows=[3])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        features, names = fe.create_rolling_features(values, "test", [3])

        assert len(names) > 0
        assert "test_rolling_mean_3" in names
        # Features are created for each window
        assert len(features) > 0

    def test_create_rolling_features_generates_stats(self):
        """Test that rolling features generates expected statistics."""
        fe = FeatureEngineer(rolling_windows=[3])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        features, names = fe.create_rolling_features(values, "speed", [3])

        # Should have rolling mean, std, max, min for window 3
        assert any("rolling_mean" in n for n in names)
        assert any("rolling_std" in n for n in names)


class TestPolynomialFeatures:
    """Tests for polynomial feature creation."""

    def test_create_polynomial_features(self):
        """Test polynomial feature creation."""
        from solar_regatta.ml import create_polynomial_features

        X = np.array([[1, 2], [3, 4], [5, 6]])

        X_poly = create_polynomial_features(X, degree=2)

        # Polynomial features should have more columns
        assert X_poly.shape[0] == X.shape[0]
        assert X_poly.shape[1] >= X.shape[1]

    def test_polynomial_degree_1_same_as_input(self):
        """Test degree 1 returns similar to input."""
        from solar_regatta.ml import create_polynomial_features

        X = np.array([[1, 2], [3, 4]])

        X_poly = create_polynomial_features(X, degree=1)

        # Degree 1 should include at least the original features
        assert X_poly.shape[1] >= X.shape[1]
