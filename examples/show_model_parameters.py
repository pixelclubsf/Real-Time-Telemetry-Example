#!/usr/bin/env python3
"""
Demo script showing model parameter tables.

Usage:
    python examples/show_model_parameters.py
"""

from solar_regatta import generate_sample_vesc_data, calculate_speeds
from solar_regatta.ml import (
    train_speed_model,
    prepare_training_data,
    print_parameter_table,
    get_model_summary,
    FeatureEngineer,
)

print("=" * 70)
print("SOLAR REGATTA - MODEL PARAMETER INSPECTOR")
print("=" * 70)
print()

# Generate sample data
print("Generating sample telemetry data...")
gps, timestamps, speeds_raw, voltage, current = \
    generate_sample_vesc_data(duration_seconds=300, interval=5)

speeds = calculate_speeds(gps, timestamps)
print(f"Generated {len(speeds)} data points")
print()

# ============================================================================
# 1. LINEAR REGRESSION MODEL (Baseline)
# ============================================================================
print("\n" + "=" * 70)
print("1. LINEAR REGRESSION MODEL (Baseline)")
print("=" * 70)

model = train_speed_model(speeds, voltage, current, timestamps)
X, y, _ = prepare_training_data(speeds, voltage, current, timestamps)

print_parameter_table(model, X)
print(f"Quick Summary: {get_model_summary(model)}")

# ============================================================================
# 2. LINEAR REGRESSION WITH FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("2. LINEAR REGRESSION WITH FEATURE ENGINEERING")
print("=" * 70)

try:
    # Create feature engineer
    engineer = FeatureEngineer(
        rolling_windows=[3, 5, 10],
        lag_features=3,
        include_derivatives=True,
        include_physics=True
    )

    # Engineer features
    X_engineered = engineer.fit_transform(
        speeds,
        voltage,
        current
    )
    y_engineered = speeds[engineer.feature_delay:]

    print(f"\nEngineered Features:")
    print(f"  Original features: 4")
    print(f"  Engineered features: {X_engineered.shape[1]}")
    print(f"  Feature names: {len(engineer.get_feature_names())} total")
    print()

    # Train model on engineered features
    from solar_regatta.ml.models import PerformanceModel
    import numpy as np

    # Fit model
    X_aug = np.column_stack([X_engineered, np.ones(len(X_engineered))])
    solution, *_ = np.linalg.lstsq(X_aug, y_engineered, rcond=None)
    coefficients = solution[:-1]
    intercept = solution[-1]

    model_engineered = PerformanceModel(
        coefficients=coefficients,
        intercept=intercept,
        feature_names=engineer.get_feature_names()
    )

    print_parameter_table(model_engineered, X_engineered)

except Exception as e:
    print(f"Error with feature engineering: {e}")

# ============================================================================
# 3. TREE-BASED MODELS (if available)
# ============================================================================
print("\n" + "=" * 70)
print("3. TREE-BASED MODELS")
print("=" * 70)

try:
    from solar_regatta.ml import RandomForestSpeedModel, XGBoostSpeedModel

    print("\n--- Random Forest Model ---")
    rf_model = RandomForestSpeedModel(n_estimators=50, max_depth=10)
    rf_model.fit(X, y)
    print_parameter_table(rf_model, X)

    print("\n--- XGBoost Model ---")
    xgb_model = XGBoostSpeedModel(n_estimators=50, max_depth=6)
    xgb_model.fit(X, y, verbose=False)
    print_parameter_table(xgb_model, X)

except ImportError:
    print("\nTree-based models not available.")
    print("Install with: pip install -e \".[ml-advanced]\"")
    print()

print("\n" + "=" * 70)
print("PARAMETER INSPECTION COMPLETE")
print("=" * 70)
