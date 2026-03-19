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
import optuna
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV

from .feature_engineering import FEATURE_COLUMNS

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

XGBOOST_PARAMS: dict = {
    "n_estimators": 1200,
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
IMPORTANCE_ESTIMATORS_CAP = 300
IMPORTANCE_ESTIMATORS_BASE = 200

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
        early_stopping_rounds: int = 100,
        variant: str | None = None,
        target_column: str = "target",
        importance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        val_gap: int = 0,
        n_bag_models: int = 3,
    ) -> None:
        self.symbol = symbol
        self.model_dir = model_dir
        self.params = params if params is not None else XGBOOST_PARAMS.copy()
        self.top_features = top_features
        self.early_stopping_rounds = early_stopping_rounds
        self.variant = variant
        self.target_column = target_column
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.val_gap = val_gap
        self.n_bag_models = n_bag_models
        self._model: XGBClassifier | None = None
        self._feature_columns: list[str] | None = None
        self._bag_models: list[XGBClassifier] = []

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

    def _drop_correlated(
        self, df: pd.DataFrame, columns: list[str], threshold: float
    ) -> list[str]:
        """Remove highly correlated features to reduce redundancy."""
        if not columns:
            return columns
        corr_matrix = df[columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
        to_drop = {
            column
            for column in upper_tri.columns
            if any(upper_tri[column].fillna(0) > threshold)
        }
        return [c for c in columns if c not in to_drop]

    def _rfecv_features(
        self, df: pd.DataFrame, columns: list[str], params: dict, n_splits: int
    ) -> list[str]:
        """Recursive feature elimination with time-series CV."""
        if len(columns) <= 2:
            return columns
        estimator = XGBClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=tscv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            rfecv.fit(df[columns].values, df[self.target_column].values)
        mask = getattr(rfecv, "support_", None)
        if mask is None or not mask.any():
            return columns
        return [col for col, keep in zip(columns, mask) if keep]

    def _prune_with_importance(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        params: dict,
        threshold: float,
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """Use XGBoost feature importances to drop uninformative predictors."""
        quick_params = params.copy()
        quick_params["n_estimators"] = min(
            quick_params.get("n_estimators", IMPORTANCE_ESTIMATORS_BASE),
            IMPORTANCE_ESTIMATORS_CAP,
        )
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

    def _walk_forward_splits(self, n_samples: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Walk-forward validation splits with fixed validation window."""
        if n_samples < 2:
            return []
        val_size = max(MIN_VAL_SAMPLES, int(n_samples * MIN_VAL_RATIO))
        max_allowed = max(int(n_samples * MAX_VAL_RATIO), 1)
        val_size = min(val_size, max_allowed)

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(n_splits):
            end_val = n_samples - (n_splits - fold - 1) * val_size
            start_val = end_val - val_size
            if start_val <= 0 or start_val >= end_val:
                continue
            gap = max(int(self.val_gap), 0)
            if start_val <= gap:
                continue
            train_end = max(start_val - gap, 0)
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(start_val, end_val)
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            splits.append((train_idx, val_idx))
        return splits

    def _cross_val_auc(
        self,
        df: pd.DataFrame,
        candidate_columns: list[str],
        params: dict,
        n_splits: int,
        verbose: bool,
        label: str | None = None,
        threshold: float | None = None,
    ) -> tuple[float, list[float]]:
        splits = self._walk_forward_splits(len(df), n_splits)
        if not splits:
            return float("nan"), []

        oof_aucs: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            selected = self._select_top_features(train_df, candidate_columns)
            if threshold is not None:
                selected, _ = self._prune_with_importance(
                    train_df, selected, params, threshold=threshold
                )

            X_tr = train_df[selected].values
            X_val = val_df[selected].values
            y_tr = train_df[self.target_column].values
            y_val = val_df[self.target_column].values

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
                print(f"  {prefix} fold {fold + 1}/{len(splits)}  AUC={auc:.4f}")

        return float(np.mean(oof_aucs)), oof_aucs

    def _bayes_optimize(
        self,
        df: pd.DataFrame,
        columns: list[str],
        n_splits: int,
        n_trials: int = 20,
        timeout: int | None = None,
        base_params: dict | None = None,
    ) -> dict:
        """Bayesian hyper-parameter search via Optuna."""

        def objective(trial: optuna.Trial) -> float:
            params = (base_params or self.params).copy()
            params.update(
                {
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
                    "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "n_estimators": trial.suggest_int("n_estimators", 400, 1600),
                }
            )
            mean_auc, _ = self._cross_val_auc(
                df,
                columns,
                params,
                n_splits=n_splits,
                verbose=False,
                label="bayes",
                threshold=self.importance_threshold,
            )
            return mean_auc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return {**self.params, **study.best_params}

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
        bayes_trials: int = 15,
    ) -> dict:
        """
        Train on a feature DataFrame that contains a target column.

        Uses walk-forward cross-validation to report out-of-fold AUC before
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
        available = self._drop_correlated(df, available, self.correlation_threshold)
        y = df[self.target_column].values

        # Handle class imbalance by up-weighting the minority class.
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        if pos + neg == 0:
            scale_pos_weight = 1.0
        elif pos == 0:
            scale_pos_weight = 1.0
        else:
            scale_pos_weight = neg / pos

        base_params = self.params.copy()
        base_params["scale_pos_weight"] = scale_pos_weight
        base_candidates = param_grid if param_grid else [base_params]
        # Bayesian optimization to enrich candidate pool
        if bayes_trials and bayes_trials > 0:
            try:
                bayes_params = self._bayes_optimize(
                    df, available, n_splits=n_splits, n_trials=bayes_trials, base_params=base_params
                )
                base_candidates.append(bayes_params)
            except Exception as exc:
                if verbose:
                    print(f"  [{self.symbol}] Optuna search failed: {exc}")
        candidates = []
        for params in base_candidates:
            candidate = params.copy()
            candidate.setdefault("scale_pos_weight", scale_pos_weight)
            candidates.append(candidate)
        cv_results: list[dict] = []
        best_idx = 0
        best_mean_auc = -np.inf

        for idx, params in enumerate(candidates):
            label = f"grid#{idx + 1}" if len(candidates) > 1 else None
            mean_auc, fold_aucs = self._cross_val_auc(
                df,
                available,
                params,
                n_splits=n_splits,
                verbose=verbose,
                label=label,
                threshold=threshold,
            )
            cv_results.append({"params": params, "mean_auc": mean_auc, "folds": fold_aucs})
            score = mean_auc if not np.isnan(mean_auc) else -np.inf
            if score > best_mean_auc:
                best_idx = idx
                best_mean_auc = score

        self.params = candidates[best_idx]
        mean_auc = best_mean_auc
        best_folds = cv_results[best_idx]["folds"]
        if verbose and len(candidates) > 1:
            print(
                f"  [{self.symbol}] best grid={best_idx + 1}/{len(candidates)} "
                f"mean OOF AUC={mean_auc:.4f}"
            )

        if verbose:
            print(f"  [{self.symbol}] mean OOF AUC = {mean_auc:.4f}")

        # Determine feature set on full data for final training
        selected = self._select_top_features(df, available)
        if len(selected) > 2:
            selected = self._drop_correlated(df, selected, self.correlation_threshold)
        if len(selected) > 2:
            selected = self._rfecv_features(
                df, selected, self.params, n_splits=n_splits
            )
        importance_ranking: list[tuple[str, float]] = []
        if threshold is not None:
            selected, importance_ranking = self._prune_with_importance(
                df, selected, self.params, threshold=threshold
            )
            if verbose:
                print(
                    f"  [{self.symbol}] importance pruning kept {len(selected)}/{len(available)} features"
                )

        X = df[selected].values
        y = df[self.target_column].values

        # Re-train on the full data set
        # Hold out a modest validation slice: at least MIN_VAL_SAMPLES or
        # MIN_VAL_RATIO of the data, but never more than MAX_VAL_RATIO so most
        # history remains for training.
        min_required = max(MIN_VAL_SAMPLES, int(len(X) * MIN_VAL_RATIO))
        max_allowed = max(int(len(X) * MAX_VAL_RATIO), 1)
        val_size = min(min_required, max_allowed)
        if len(X) - val_size < 1:
            val_size = max(1, len(X) - 1)
        gap = max(int(self.val_gap), 0)
        val_start = len(X) - val_size
        desired_train_end = val_start - gap
        min_train = max(50, val_size, 1)
        max_train = max(val_start - 1, 1)
        # Clamp train_end to enforce gap (no overlap), minimum train size, and bounds
        train_end = min(max(desired_train_end, min_train), max_train)
        X_train, X_val = X[:train_end], X[val_start:]
        y_train, y_val = y[:train_end], y[val_start:]

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

        # Bagging ensemble with different seeds (size = n_bag_models); base random_state is defined in XGBOOST_PARAMS
        self._bag_models = []
        for seed in range(self.n_bag_models):
            bag_params = self.params.copy()
            bag_params["random_state"] = bag_params.get("random_state", 42) + seed
            bag_model = XGBClassifier(**bag_params)
            bag_model.fit(X_train, y_train, **fit_kwargs)
            self._bag_models.append(bag_model)
        # Keep the first model for backward compatibility with save/load APIs
        self._model = self._bag_models[0]

        self._feature_columns = selected

        train_accuracy: float | None = None
        test_accuracy: float | None = None
        train_precision: float | None = None
        train_recall: float | None = None
        train_f1: float | None = None
        test_precision: float | None = None
        test_recall: float | None = None
        test_f1: float | None = None

        def _compute_prf(true, pred):
            precision, recall, f1, _ = precision_recall_fscore_support(
                true,
                pred,
                average="binary",
                zero_division=0,
            )
            return float(precision), float(recall), float(f1)
        if len(X_train) > 0:
            train_pred = self._model.predict(X_train)
            train_accuracy = float(accuracy_score(y_train, train_pred))
            train_precision, train_recall, train_f1 = _compute_prf(y_train, train_pred)
        if len(X_val) > 0:
            test_pred = self._model.predict(X_val)
            test_accuracy = float(accuracy_score(y_val, test_pred))
            test_precision, test_recall, test_f1 = _compute_prf(y_val, test_pred)

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
                acc_msg.append(
                    f"train acc={train_accuracy:.4f} "
                    f"prec={train_precision:.4f} rec={train_recall:.4f} f1={train_f1:.4f}"
                )
            if test_accuracy is not None:
                acc_msg.append(
                    f"test acc={test_accuracy:.4f} "
                    f"prec={test_precision:.4f} rec={test_recall:.4f} f1={test_f1:.4f}"
                )
            if acc_msg:
                print(f"  [{self.symbol}] " + "  ".join(acc_msg))

        return {
            "oof_auc": mean_auc,
            "fold_aucs": best_folds,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
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
        models = self._bag_models or [self._model]
        probas = [m.predict_proba(X)[:, 1] for m in models]
        return np.mean(probas, axis=0)

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
            "bag_models": self._bag_models,
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
        self._bag_models = payload.get("bag_models", [self._model])
        self._feature_columns = payload["feature_columns"]
        self.params = payload.get("params", self.params)
        self.variant = payload.get("variant", self.variant)
        self.target_column = payload.get("target_column", self.target_column)
        return self
