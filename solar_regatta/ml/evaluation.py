"""Model evaluation and cross-validation utilities."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def time_series_split(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Time-series aware train/test splits.

    Args:
        X: Feature matrix
        y: Target vector
        n_splits: Number of splits
        test_size: Test set size (if None, uses expanding window)

    Returns:
        List of (X_train, X_test, y_train, y_test) tuples
    """
    n_samples = len(X)
    splits = []

    if test_size is None:
        # Expanding window
        step = n_samples // (n_splits + 1)
        for i in range(n_splits):
            train_end = step * (i + 2)
            test_end = min(train_end + step, n_samples)

            X_train, X_test = X[:train_end], X[train_end:test_end]
            y_train, y_test = y[:train_end], y[train_end:test_end]

            if len(X_test) > 0:
                splits.append((X_train, X_test, y_train, y_test))
    else:
        # Rolling window with fixed test size
        for i in range(n_splits):
            test_start = n_samples - test_size * (n_splits - i)
            test_end = test_start + test_size

            if test_start > 0:
                X_train, X_test = X[:test_start], X[test_start:test_end]
                y_train, y_test = y[:test_start], y[test_start:test_end]
                splits.append((X_train, X_test, y_train, y_test))

    return splits


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate model performance.

    Args:
        model: Fitted model with predict() method
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary of metrics
    """
    predictions = model.predict(X_test)
    residuals = y_test - predictions

    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(mse))

    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Additional metrics
    mape = float(np.mean(np.abs(residuals / (y_test + 1e-8)))) * 100
    max_error = float(np.max(np.abs(residuals)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "max_error": max_error,
    }


def cross_validate(
    model_class: type,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    **model_kwargs
) -> Dict[str, Any]:
    """Perform time-series cross-validation.

    Args:
        model_class: Model class to instantiate
        X: Feature matrix
        y: Target vector
        n_splits: Number of CV splits
        **model_kwargs: Arguments for model initialization

    Returns:
        Cross-validation results
    """
    splits = time_series_split(X, y, n_splits=n_splits)
    scores = []

    for X_train, X_test, y_train, y_test in splits:
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        scores.append(metrics)

    # Average metrics across folds
    avg_metrics = {}
    for key in scores[0].keys():
        values = [s[key] for s in scores]
        avg_metrics[f"{key}_mean"] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)

    return avg_metrics


def compare_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models using cross-validation.

    Args:
        models: Dict of {name: (model_class, kwargs)}
        X: Feature matrix
        y: Target vector
        n_splits: Number of CV splits

    Returns:
        Comparison results for each model
    """
    results = {}

    for name, (model_class, kwargs) in models.items():
        print(f"Evaluating {name}...")
        cv_results = cross_validate(model_class, X, y, n_splits, **kwargs)
        results[name] = cv_results

    return results
