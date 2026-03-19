"""
XGBoost wrapper for the crypto trend prediction model.

Each symbol gets its own trained XGBoostClassifier. Models are saved to /
loaded from disk using joblib so that predictions can be served without
re-training.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

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
        return keep if keep else candidate_columns

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
            ``{"oof_auc": float}`` — average out-of-fold ROC-AUC score.
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
            fold_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
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
        split_idx = int(len(X) * 0.8)
        split_idx = min(max(split_idx, 1), len(X) - 1)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self._model = XGBClassifier(**self.params)
        eval_set = [(X_val, y_val)] if len(X_val) > 0 else []
        fit_kwargs = {
            "eval_set": eval_set,
            "verbose": False,
        }
        if eval_set:
            fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._feature_columns = selected

        return {"oof_auc": mean_auc}

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
            raise RuntimeError("Model metadata missing feature columns.")
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
