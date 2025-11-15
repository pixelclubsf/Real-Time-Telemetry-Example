"""Tree-based models for speed prediction."""
from __future__ import annotations

from typing import Dict, Optional, Any

import numpy as np

# Conditionally import tree-based libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class RandomForestSpeedModel:
    """Random Forest model for speed prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            random_state: Random seed
            **kwargs: Additional sklearn RandomForest parameters
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            )

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            **kwargs
        )
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model.

        Args:
            X: Feature matrix
            y: Target values
        """
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances.

        Returns:
            Feature importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted first")
        return self.feature_importances_


class XGBoostSpeedModel:
    """XGBoost model for speed prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
            **kwargs: Additional XGBoost parameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required. Install with: pip install xgboost"
            )

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ):
        """Fit the model.

        Args:
            X: Feature matrix
            y: Target values
            eval_set: Validation set for early stopping
            early_stopping_rounds: Number of rounds for early stopping
            verbose: Print training progress
        """
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        if not verbose:
            fit_params['verbose'] = False

        self.model.fit(X, y, **fit_params)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        return self.model.predict(X)

    def get_feature_importance(self, importance_type: str = 'weight') -> np.ndarray:
        """Get feature importances.

        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')

        Returns:
            Feature importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted first")
        return self.feature_importances_


class LightGBMSpeedModel:
    """LightGBM model for speed prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize LightGBM model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Learning rate
            num_leaves: Maximum tree leaves
            random_state: Random seed
            **kwargs: Additional LightGBM parameters
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is required. Install with: pip install lightgbm"
            )

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            verbose=-1,
            **kwargs
        )
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = None,
    ):
        """Fit the model.

        Args:
            X: Feature matrix
            y: Target values
            eval_set: Validation set for early stopping
            early_stopping_rounds: Number of rounds for early stopping
        """
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds is not None:
            fit_params['callbacks'] = [
                lgb.early_stopping(early_stopping_rounds, verbose=False)
            ]

        self.model.fit(X, y, **fit_params)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        return self.model.predict(X)

    def get_feature_importance(self, importance_type: str = 'split') -> np.ndarray:
        """Get feature importances.

        Args:
            importance_type: Type of importance ('split' or 'gain')

        Returns:
            Feature importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted first")
        return self.feature_importances_


class GradientBoostingSpeedModel:
    """Scikit-learn Gradient Boosting model for speed prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs
    ):
        """Initialize Gradient Boosting model.

        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
            **kwargs: Additional sklearn GradientBoosting parameters
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            )

        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model.

        Args:
            X: Feature matrix
            y: Target values
        """
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances.

        Returns:
            Feature importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted first")
        return self.feature_importances_
