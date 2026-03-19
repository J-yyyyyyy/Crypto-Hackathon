"""
XGBoost wrapper for the crypto trend prediction model.

Each symbol gets its own trained XGBoostClassifier. Models are saved to /
loaded from disk using joblib so that predictions can be served without
re-training.
"""

from __future__ import annotations

import os
import warnings
from inspect import signature
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from .feature_engineering import FEATURE_COLUMNS

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

XGBOOST_PARAMS: dict = {
    "n_estimators": 800,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 4,
    "gamma": 1.0,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}

MIN_VAL_SAMPLES = 10
MIN_VAL_RATIO = 0.1
MAX_VAL_RATIO = 0.2

_FIT_PARAMS = signature(XGBClassifier.fit).parameters
SUPPORTS_CALLBACKS = "callbacks" in _FIT_PARAMS
SUPPORTS_EARLY_STOPPING = "early_stopping_rounds" in _FIT_PARAMS


class CryptoTrendModel:
    """
    A thin wrapper around ``XGBClassifier`` that adds cross-validation
    reporting, persistence helpers and a clean predict API.

    Parameters
    ----------
    symbol : str
        The trading-pair symbol this model is trained for.
    model_dir : str
        Directory used for saving / loading model files.
    params : dict | None
        XGBoost hyper-parameters.  Defaults to ``XGBOOST_PARAMS``.
    variant : str | None
        Optional variant tag used to distinguish multiple targets per symbol.
    target_column : str
        Name of the target column inside the feature DataFrame.
    """

    def __init__(
        self,
        symbol: str,
        model_dir: str = DEFAULT_MODEL_DIR,
        params: dict | None = None,
        top_features: int = 20,
        early_stopping_rounds: int = 50,
        variant: str | None = None,
        target_column: str = "target",
        importance_threshold: float = 0.0,
    ) -> None:
        self.symbol = symbol
        self.model_dir = model_dir
        self.params = params if params is not None else XGBOOST_PARAMS.copy()
        self.top_features = top_features
        self.early_stopping_rounds = early_stopping_rounds
        self.variant = variant
        self.target_column = target_column
        self.importance_threshold = importance_threshold
        self._model: XGBClassifier | None = None
        self._feature_columns: list[str] | None = None

    def _select_top_features(
        self, df: pd.DataFrame, candidate_columns: list[str]
    ) -> list[str]:
        """Select most stable predictors via correlation with the target."""
        corr = (
            df[candidate_columns + [self.target_column]]
            .corr()[self.target_column]
            .drop(self.target_column)
            .abs()
            .dropna()
        )
        ranked = corr.sort_values(ascending=False)
        keep = ranked.head(self.top_features).index.tolist()
        if keep:
            return keep
        warnings.warn(
            f"[{self.symbol}] No usable correlations for feature selection; "
            f"using all {len(candidate_columns)} features."
        )
        return candidate_columns

    def _prune_with_importance(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        params: dict,
        threshold: float,
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """Use XGBoost feature importances to drop uninformative predictors."""
        quick_params = params.copy()
        quick_params["n_estimators"] = min(quick_params.get("n_estimators", 200), 300)
        model = XGBClassifier(**quick_params)
        model.fit(
            df[feature_columns].values,
            df[self.target_column].values,
            verbose=False,
        )
        importances = model.feature_importances_
        ranked = sorted(
            zip(feature_columns, importances), key=lambda kv: kv[1], reverse=True
        )
        keep = [feat for feat, score in ranked if score > threshold]
        if not keep:
            keep = feature_columns
        return keep, ranked

    def _cross_val_auc(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict,
        n_splits: int,
        verbose: bool,
        label: str | None = None,
    ) -> tuple[float, list[float]]:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_aucs: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = XGBClassifier(**params)
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }
            if self.early_stopping_rounds and len(y_val) > 0:
                if SUPPORTS_CALLBACKS:
                    fit_kwargs["callbacks"] = [
                        EarlyStopping(
                            rounds=self.early_stopping_rounds,
                            save_best=True,
                            metric_name="auc",
                        )
                    ]
                elif SUPPORTS_EARLY_STOPPING:
                    fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            fold_model.fit(
                X_tr,
                y_tr,
                **fit_kwargs,
            )
            proba = fold_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            oof_aucs.append(auc)
            if verbose:
                prefix = f"[{self.symbol}]"
                if label:
                    prefix += f" {label}"
                print(f"  {prefix} fold {fold + 1}/{n_splits}  AUC={auc:.4f}")

        return float(np.mean(oof_aucs)), oof_aucs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        verbose: bool = True,
        target_column: str | None = None,
        param_grid: list[dict] | None = None,
        importance_threshold: float | None = None,
    ) -> dict:
        """
        Train on a feature DataFrame that contains a target column.

        Uses time-series cross-validation to report out-of-fold AUC before
        fitting the final model on all available data. Optionally runs a
        small hyper-parameter grid search and prunes uninformative features
        using XGBoost feature importances.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with ``FEATURE_COLUMNS`` columns and a target column.
        n_splits : int
            Number of folds for ``TimeSeriesSplit``.
        verbose : bool
            Whether to print per-fold metrics.
        target_column : str | None
            Which target column to use (defaults to current ``self.target_column``).
        param_grid : list[dict] | None
            Optional grid of parameter dicts to search; best mean OOF AUC wins.
        importance_threshold : float | None
            Drop features whose importance is at or below this threshold after
            a quick importance-only fit. Defaults to ``self.importance_threshold``.

        Returns
        -------
        dict
            Metrics including mean out-of-fold ROC-AUC, train/test accuracy,
            and the chosen parameter set.
        """
        if target_column:
            self.target_column = target_column
        threshold = (
            importance_threshold
            if importance_threshold is not None
            else self.importance_threshold
        )

        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        selected = self._select_top_features(df, available)
        X = df[selected].values
        y = df[self.target_column].values

        candidates = param_grid if param_grid else [self.params]
        cv_results: list[dict] = []
        best_idx = 0
        best_mean_auc = -np.inf

        for idx, params in enumerate(candidates):
            label = f"grid#{idx + 1}" if len(candidates) > 1 else None
            mean_auc, fold_aucs = self._cross_val_auc(
                X, y, params, n_splits=n_splits, verbose=verbose, label=label
            )
            cv_results.append({"params": params, "mean_auc": mean_auc, "folds": fold_aucs})
            if mean_auc > best_mean_auc:
                best_idx = idx
                best_mean_auc = mean_auc

        self.params = candidates[best_idx]
        mean_auc = best_mean_auc
        best_folds = cv_results[best_idx]["folds"]
        if verbose and len(candidates) > 1:
            print(
                f"  [{self.symbol}] best grid={best_idx + 1}/{len(candidates)} "
                f"mean OOF AUC={mean_auc:.4f}"
            )

        importance_ranking: list[tuple[str, float]] = []
        if threshold is not None:
            pruned, importance_ranking = self._prune_with_importance(
                df, selected, self.params, threshold=threshold
            )
            if pruned != selected and verbose:
                print(
                    f"  [{self.symbol}] importance pruning kept {len(pruned)}/{len(selected)} features"
                )
            selected = pruned
            X = df[selected].values
            mean_auc, best_folds = self._cross_val_auc(
                X, y, self.params, n_splits=n_splits, verbose=verbose, label="pruned"
            )

        if verbose:
            print(f"  [{self.symbol}] mean OOF AUC = {mean_auc:.4f}")

        # Re-train on the full data set
        # Hold out a modest validation slice: at least MIN_VAL_SAMPLES or
        # MIN_VAL_RATIO of the data, but never more than MAX_VAL_RATIO so most
        # history remains for training.
        min_required = max(MIN_VAL_SAMPLES, int(len(X) * MIN_VAL_RATIO))
        max_allowed = max(int(len(X) * MAX_VAL_RATIO), 1)
        val_size = min(min_required, max_allowed)
        if len(X) - val_size < 1:
            val_size = max(1, len(X) - 1)
        split_idx = len(X) - val_size
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self._model = XGBClassifier(**self.params)
        eval_set = [(X_val, y_val)] if len(X_val) > 0 else []
        fit_kwargs = {
            "eval_set": eval_set,
            "verbose": False,
        }
        if eval_set and self.early_stopping_rounds:
            if SUPPORTS_CALLBACKS:
                fit_kwargs["callbacks"] = [
                    EarlyStopping(
                        rounds=self.early_stopping_rounds,
                        save_best=True,
                        metric_name="auc",
                    )
                ]
            elif SUPPORTS_EARLY_STOPPING:
                fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._feature_columns = selected

        train_accuracy: float | None = None
        test_accuracy: float | None = None
        if len(X_train) > 0:
            train_pred = self._model.predict(X_train)
            train_accuracy = float(accuracy_score(y_train, train_pred))
        if len(X_val) > 0:
            test_pred = self._model.predict(X_val)
            test_accuracy = float(accuracy_score(y_val, test_pred))

        feature_importance = []
        if hasattr(self._model, "feature_importances_"):
            feature_importance = sorted(
                zip(selected, self._model.feature_importances_),
                key=lambda kv: kv[1],
                reverse=True,
            )

        if verbose:
            acc_msg = []
            if train_accuracy is not None:
                acc_msg.append(f"train acc={train_accuracy:.4f}")
            if test_accuracy is not None:
                acc_msg.append(f"test acc={test_accuracy:.4f}")
            if acc_msg:
                print(f"  [{self.symbol}] " + "  ".join(acc_msg))

        return {
            "oof_auc": mean_auc,
            "fold_aucs": best_folds,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "params": self.params,
            "features": selected,
            "importance_ranking": importance_ranking or feature_importance,
            "target_column": self.target_column,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return the probability of an upward price move for each row in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the same feature columns used during training.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Probability of the positive class (price goes up).
        """
        if self._model is None:
            raise RuntimeError(
                f"Model for {self.symbol} is not trained. "
                "Call train() or load() first."
            )
        if not self._feature_columns:
            raise RuntimeError(
                "Model feature columns not set or empty. "
                "Ensure train() or load() has completed successfully."
            )
        available = [c for c in self._feature_columns if c in df.columns]
        X = df[available].values
        return self._model.predict_proba(X)[:, 1]

    def predict_latest(self, df: pd.DataFrame) -> float:
        """
        Predict the probability of a price rise using only the most recent row.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame (the last row is used).

        Returns
        -------
        float
            Probability in [0, 1].
        """
        proba = self.predict_proba(df.iloc[[-1]])
        return float(proba[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _model_path(self) -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        suffix = f"_{self.variant}" if self.variant else ""
        return os.path.join(self.model_dir, f"{self.symbol}{suffix}.joblib")

    def save(self) -> str:
        """Persist model to disk. Returns the saved file path."""
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        payload = {
            "model": self._model,
            "feature_columns": self._feature_columns,
            "symbol": self.symbol,
            "params": self.params,
            "variant": self.variant,
            "target_column": self.target_column,
        }
        path = self._model_path()
        joblib.dump(payload, path)
        return path

    def load(self) -> "CryptoTrendModel":
        """Load a previously saved model from disk. Returns self."""
        path = self._model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved model found for {self.symbol} at {path}. "
                "Run train.py first."
            )
        payload = joblib.load(path)
        self._model = payload["model"]
        self._feature_columns = payload["feature_columns"]
        self.params = payload.get("params", self.params)
        self.variant = payload.get("variant", self.variant)
        self.target_column = payload.get("target_column", self.target_column)
        return self
