"""Model parameter inspection and display utilities."""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


def count_model_parameters(model: Any) -> Dict[str, Any]:
    """Count parameters in a model.

    Args:
        model: Model instance (PerformanceModel, RandomForest, XGBoost, etc.)

    Returns:
        Dictionary with parameter counts and details
    """
    from solar_regatta.ml.models import PerformanceModel

    info = {
        "model_type": type(model).__name__,
        "total_parameters": 0,
        "details": {}
    }

    # Linear regression model
    if isinstance(model, PerformanceModel):
        n_coefficients = len(model.coefficients)
        info["total_parameters"] = n_coefficients + 1  # +1 for intercept
        info["details"] = {
            "coefficients": n_coefficients,
            "intercept": 1,
            "feature_names": model.feature_names
        }

    # Tree-based models
    elif hasattr(model, 'model'):
        underlying = model.model
        model_type = type(underlying).__name__

        # Random Forest or Gradient Boosting (sklearn)
        if hasattr(underlying, 'estimators_'):
            n_trees = len(underlying.estimators_)
            # Estimate nodes per tree (approximate)
            total_nodes = 0
            if hasattr(underlying, 'estimators_'):
                for estimator in underlying.estimators_:
                    if hasattr(estimator, 'tree_'):
                        total_nodes += estimator.tree_.node_count

            info["total_parameters"] = total_nodes
            info["details"] = {
                "n_trees": n_trees,
                "total_nodes": total_nodes,
                "avg_nodes_per_tree": total_nodes / n_trees if n_trees > 0 else 0,
                "model_type": model_type
            }

        # XGBoost
        elif hasattr(underlying, 'get_booster'):
            booster = underlying.get_booster()
            n_trees = len(booster.get_dump())
            info["total_parameters"] = n_trees
            info["details"] = {
                "n_trees": n_trees,
                "n_features": underlying.n_features_in_ if hasattr(underlying, 'n_features_in_') else "unknown",
                "model_type": "XGBoost"
            }

        # LightGBM
        elif hasattr(underlying, 'booster_'):
            n_trees = underlying.booster_.current_iteration()
            info["total_parameters"] = n_trees
            info["details"] = {
                "n_trees": n_trees,
                "n_features": underlying.n_features_in_ if hasattr(underlying, 'n_features_in_') else "unknown",
                "model_type": "LightGBM"
            }

    return info


def print_parameter_table(model: Any, feature_matrix: Optional[np.ndarray] = None):
    """Print a formatted parameter table for the model.

    Args:
        model: Model instance
        feature_matrix: Optional feature matrix to show input dimensions
    """
    from solar_regatta.ml.models import PerformanceModel

    print("=" * 70)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 70)
    print()

    info = count_model_parameters(model)

    print(f"Model Type: {info['model_type']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print()

    # Linear regression details
    if isinstance(model, PerformanceModel):
        print("COEFFICIENTS:")
        print("-" * 70)
        print(f"{'Feature':<20} {'Coefficient':>15} {'Abs Value':>15}")
        print("-" * 70)

        for name, coef in zip(model.feature_names, model.coefficients):
            print(f"{name:<20} {coef:>15.6f} {abs(coef):>15.6f}")

        print("-" * 70)
        print(f"{'Intercept':<20} {model.intercept:>15.6f} {abs(model.intercept):>15.6f}")
        print()

        # Feature importance (by absolute coefficient value)
        importance_order = np.argsort(np.abs(model.coefficients))[::-1]
        print("FEATURE IMPORTANCE (by absolute coefficient):")
        print("-" * 70)
        print(f"{'Rank':<6} {'Feature':<20} {'Importance':>15}")
        print("-" * 70)
        for rank, idx in enumerate(importance_order, 1):
            print(f"{rank:<6} {model.feature_names[idx]:<20} {abs(model.coefficients[idx]):>15.6f}")
        print()

    # Tree-based model details
    elif "details" in info and info["details"]:
        details = info["details"]
        print("MODEL CONFIGURATION:")
        print("-" * 70)

        if "n_trees" in details:
            print(f"Number of Trees: {details['n_trees']:,}")

        if "total_nodes" in details:
            print(f"Total Nodes: {details['total_nodes']:,}")
            print(f"Average Nodes per Tree: {details['avg_nodes_per_tree']:.1f}")

        if "n_features" in details:
            print(f"Number of Features: {details['n_features']}")

        print()

        # Feature importance for tree models
        if hasattr(model, 'get_feature_importance'):
            try:
                importance = model.get_feature_importance()
                importance_order = np.argsort(importance)[::-1]

                print("FEATURE IMPORTANCE:")
                print("-" * 70)
                print(f"{'Rank':<6} {'Feature Index':<15} {'Importance':>15}")
                print("-" * 70)
                for rank, idx in enumerate(importance_order[:10], 1):  # Top 10
                    print(f"{rank:<6} {idx:<15} {importance[idx]:>15.6f}")
                print()
            except:
                pass

    # Input dimensions
    if feature_matrix is not None:
        print("INPUT DIMENSIONS:")
        print("-" * 70)
        print(f"Training Samples: {feature_matrix.shape[0]:,}")
        print(f"Features per Sample: {feature_matrix.shape[1]:,}")
        print(f"Total Input Values: {feature_matrix.size:,}")
        print()

    print("=" * 70)


def get_model_summary(model: Any) -> str:
    """Get a one-line summary of model parameters.

    Args:
        model: Model instance

    Returns:
        String summary
    """
    info = count_model_parameters(model)

    if "n_trees" in info["details"]:
        return f"{info['model_type']}: {info['details']['n_trees']} trees"
    else:
        return f"{info['model_type']}: {info['total_parameters']} parameters"
