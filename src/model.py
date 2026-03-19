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
    """

    def __init__(
        self,
        symbol: str,
        model_dir: str = DEFAULT_MODEL_DIR,
        params: dict | None = None,
        top_features: int = 20,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.symbol = symbol
        self.model_dir = model_dir
        self.params = params if params is not None else XGBOOST_PARAMS.copy()
        self.top_features = top_features
        self.early_stopping_rounds = early_stopping_rounds
        self._model: XGBClassifier | None = None
        self._feature_columns: list[str] | None = None

    def _select_top_features(
        self, df: pd.DataFrame, candidate_columns: list[str]
    ) -> list[str]:
        """Select most stable predictors via correlation with the target."""
        corr = (
            df[candidate_columns + ["target"]]
            .corr()["target"]
            .drop("target")
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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Train on a feature DataFrame that contains a ``target`` column.

        Uses time-series cross-validation to report out-of-fold AUC before
        fitting the final model on all available data.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with ``FEATURE_COLUMNS`` columns and a ``target`` column.
        n_splits : int
            Number of folds for ``TimeSeriesSplit``.
        verbose : bool
            Whether to print per-fold metrics.

        Returns
        -------
        dict
            Metrics including mean out-of-fold ROC-AUC and train/test accuracy.
        """
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        selected = self._select_top_features(df, available)
        X = df[selected].values
        y = df["target"].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_aucs: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = XGBClassifier(**self.params)
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
                print(f"  [{self.symbol}] fold {fold + 1}/{n_splits}  AUC={auc:.4f}")

        mean_auc = float(np.mean(oof_aucs))
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
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
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
        return os.path.join(self.model_dir, f"{self.symbol}.joblib")

    def save(self) -> str:
        """Persist model to disk. Returns the saved file path."""
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        payload = {
            "model": self._model,
            "feature_columns": self._feature_columns,
            "symbol": self.symbol,
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
        return self
