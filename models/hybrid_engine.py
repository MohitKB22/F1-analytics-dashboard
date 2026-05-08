"""
models/hybrid_engine.py
════════════════════════════════════════════════════════════════
Hybrid Prediction Engine: XGBoost × LSTM × Weather (1950–2026)

Final Score = (w1 × XGBoost) + (w2 × LSTM) + (w3 × Weather)
════════════════════════════════════════════════════════════════
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path regardless of how this file is invoked
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.xgboost_model import XGBoostRacePredictor
from models.lstm_model import get_lstm_model
from data.features import (
    FeaturePipeline, compute_weather_impact_score,
    add_performance_features, weather_impact,
)


@dataclass
class HybridConfig:
    w_xgboost: float = 0.50
    w_lstm:    float = 0.30
    w_weather: float = 0.20
    sequence_length: int = 5
    model_dir: str = "models/saved"

    def __post_init__(self):
        total = self.w_xgboost + self.w_lstm + self.w_weather
        if abs(total - 1.0) > 1e-6:
            self.w_xgboost /= total
            self.w_lstm    /= total
            self.w_weather /= total


@dataclass
class PredictionResult:
    driver: str
    constructor: str
    predicted_position: int
    win_probability: float
    xgb_score: float
    lstm_score: float
    weather_adjusted_score: float
    weather_impact: float
    confidence: float


class HybridF1Engine:
    def __init__(self, config: HybridConfig = None):
        self.cfg = config or HybridConfig()
        self.xgb_model = XGBoostRacePredictor()
        self.lstm_model = get_lstm_model(seq_len=self.cfg.sequence_length)
        self.feature_pipeline = FeaturePipeline(sequence_length=self.cfg.sequence_length)
        self.is_trained = False

    def train(self, df: pd.DataFrame, verbose=True) -> dict:
        if verbose:
            print("=" * 60)
            print("  F1 Hybrid Engine — Training (1950-2026 Dataset)")
            print("=" * 60)

        if verbose: print("[1/4] Engineering features…")
        df = add_performance_features(df)
        df = weather_impact(df)

        if verbose: print("[2/4] Preparing tensors…")
        X_tab, X_seq, y = self.feature_pipeline.fit_transform(df)

        if verbose: print("[3/4] Training XGBoost…")
        self.xgb_model.fit(X_tab, y,
                           feature_names=self.feature_pipeline.tabular_cols,
                           verbose=verbose)

        if verbose: print("[4/4] Training LSTM…")
        self.lstm_model.fit(X_seq, y, epochs=60, verbose=0)

        self.is_trained = True

        xgb_preds  = self.xgb_model.predict(X_tab)
        lstm_preds = self.lstm_model.predict(X_seq)
        hybrid     = self.cfg.w_xgboost*xgb_preds + self.cfg.w_lstm*lstm_preds

        metrics = {
            "mae_xgboost": round(float(np.mean(np.abs(xgb_preds - y))),  3),
            "mae_lstm":    round(float(np.mean(np.abs(lstm_preds - y))),  3),
            "mae_hybrid":  round(float(np.mean(np.abs(hybrid - y))),      3),
            "n_samples":   int(len(y)),
            "n_tab_features": int(X_tab.shape[1]),
            "n_seq_features": int(X_seq.shape[2]),
        }
        if verbose:
            print("\n── Training Metrics ─────────────────────────────────")
            for k, v in metrics.items():
                print(f"   {k:<25}: {v}")
            print("=" * 60)
        return metrics

    def predict_race(self, race_df: pd.DataFrame,
                     weather: dict = None) -> list:
        if not self.is_trained:
            raise RuntimeError("Engine not trained.")

        weather = weather or {}
        w_score = compute_weather_impact_score(weather)
        is_wet  = weather.get("is_wet", False)

        df = add_performance_features(race_df.copy()) \
            if "driver_avg_pos_5" not in race_df.columns else race_df.copy()
        df = weather_impact(df)
        X_tab, X_seq = self.feature_pipeline.transform(df)

        xgb_scores  = self.xgb_model.predict(X_tab)
        lstm_scores = self.lstm_model.predict(X_seq)

        weather_mod = np.ones(len(df))
        if is_wet and "weather_score" in df.columns:
            ws = df["weather_score"].values
            weather_mod = 1 - 0.08 * (1 - ws / (ws.max() + 1e-9))

        hybrid = (
            self.cfg.w_xgboost * xgb_scores +
            self.cfg.w_lstm    * lstm_scores +
            self.cfg.w_weather * xgb_scores * (2 - w_score)
        ) * weather_mod

        def softmin(s):
            neg = -s; e = np.exp(neg - neg.max()); return e / e.sum()

        win_probs = softmin(hybrid)
        corr = np.corrcoef(xgb_scores, lstm_scores)[0, 1]
        agreement = max(0, float(corr))

        driver_col = "driver_name" if "driver_name" in df.columns else "driver_id"
        team_col   = "constructor_name" if "constructor_name" in df.columns else "constructor_id"

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            results.append(PredictionResult(
                driver=str(row.get(driver_col, f"Driver {i+1}")),
                constructor=str(row.get(team_col, "Unknown")),
                predicted_position=0,
                win_probability=float(win_probs[i]),
                xgb_score=float(xgb_scores[i]),
                lstm_score=float(lstm_scores[i]),
                weather_adjusted_score=float(hybrid[i]),
                weather_impact=float(w_score),
                confidence=round(agreement * win_probs[i] * 10, 4),
            ))

        results.sort(key=lambda r: r.weather_adjusted_score)
        for rank, r in enumerate(results, 1):
            r.predicted_position = rank
        return results

    def predict_constructor_championship(self, results: list) -> pd.DataFrame:
        pts_map = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
        rows = [{"constructor": r.constructor, "driver": r.driver,
                 "predicted_position": r.predicted_position,
                 "win_probability": r.win_probability} for r in results]
        df = pd.DataFrame(rows)
        df["points_estimate"] = df["predicted_position"].map(pts_map).fillna(0)
        team = (df.groupby("constructor")
                  .agg(total_points=("points_estimate","sum"),
                       avg_position=("predicted_position","mean"),
                       best_position=("predicted_position","min"))
                  .sort_values("total_points", ascending=False)
                  .reset_index())
        team["championship_rank"] = range(1, len(team)+1)
        return team

    def save(self, directory=None):
        d = directory or self.cfg.model_dir
        Path(d).mkdir(parents=True, exist_ok=True)
        self.xgb_model.save(f"{d}/xgboost_f1.pkl")
        try:
            self.lstm_model.save(f"{d}/lstm_f1.keras")
        except Exception:
            pass
        joblib.dump({"feature_pipeline": self.feature_pipeline,
                     "config": self.cfg}, f"{d}/pipeline.pkl")
        print(f"[HybridEngine] Saved → {d}/")

    @classmethod
    def load(cls, directory="models/saved"):
        meta = joblib.load(f"{directory}/pipeline.pkl")
        engine = cls(config=meta["config"])
        engine.feature_pipeline = meta["feature_pipeline"]
        engine.xgb_model = XGBoostRacePredictor.load(f"{directory}/xgboost_f1.pkl")
        try:
            from models.lstm_model import LSTMRacePredictor
            engine.lstm_model = LSTMRacePredictor.load(f"{directory}/lstm_f1.keras")
        except Exception:
            engine.lstm_model = get_lstm_model()
        engine.is_trained = True
        return engine
