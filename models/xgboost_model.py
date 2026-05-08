"""
models/xgboost_model.py
════════════════════════════════════════════════════════════════
XGBoost tabular model for F1 race position prediction.
════════════════════════════════════════════════════════════════
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb


class XGBoostRacePredictor:
    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    def __init__(self, params: dict = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_names: list = []
        self.is_fitted = False

    def fit(self, X, y, feature_names=None, eval_set=None, verbose=True):
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        fit_kwargs = {}
        if eval_set:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["verbose"] = 50 if verbose else False
        self.model.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        if verbose:
            mae = mean_absolute_error(y, self.model.predict(X))
            print(f"[XGBoost] Train MAE: {mae:.3f}")
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")
        return self.model.predict(X)

    def predict_proba_winner(self, X) -> np.ndarray:
        scores = self.predict(X)
        neg = -scores
        exp = np.exp(neg - neg.max())
        return exp / exp.sum()

    def feature_importance(self, top_n=20) -> pd.DataFrame:
        gain = self.model.get_booster().get_score(importance_type="gain")
        df = pd.DataFrame(gain.items(), columns=["feature", "importance"])
        df = df.sort_values("importance", ascending=False).head(top_n)
        df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(1)
        return df.reset_index(drop=True)

    def cross_validate(self, X, y, cv=5) -> dict:
        scores = cross_val_score(self.model, X, y,
                                 cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        return {"cv_mae_mean": round(-scores.mean(), 3), "cv_mae_std": round(scores.std(), 3)}

    def save(self, path="models/saved/xgboost_f1.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names,
                     "params": self.params}, path)
        print(f"[XGBoost] Saved → {path}")

    @classmethod
    def load(cls, path="models/saved/xgboost_f1.pkl"):
        data = joblib.load(path)
        obj = cls(params=data["params"])
        obj.model = data["model"]
        obj.feature_names = data["feature_names"]
        obj.is_fitted = True
        print(f"[XGBoost] Loaded ← {path}")
        return obj
