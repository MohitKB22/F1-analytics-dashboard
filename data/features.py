"""
data/features.py
════════════════════════════════════════════════════════════════════════
Feature engineering for the F1 Hybrid Prediction System (1950-2026).
Produces tabular features for XGBoost and sequence features for LSTM.
════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ── Weather Impact ───────────────────────────────────────────────────

def compute_weather_impact_score(weather: dict) -> float:
    """Single-number weather impact score [0,1]. 1.0 = perfect dry."""
    rain = 1 if weather.get("rainfall", 0) > 0 else 0
    grip = (1 - weather.get("humidity", 50) / 100) * (1 - rain)
    wind = min(weather.get("wind_speed", 10) / 60, 1)
    temp = min(weather.get("track_temp", 35) / 50, 0.9)
    return round(float(np.clip(0.4*grip + 0.3*(1-wind) + 0.3*temp, 0, 1)), 4)


def weather_impact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rainfall" in df.columns:
        df["rain_factor"] = (df["rainfall"] > 0).astype(int)
    else:
        df["rain_factor"] = 0
    if "track_temp" in df.columns:
        df["temp_effect"] = df["track_temp"] / 50.0
    else:
        df["temp_effect"] = 0.7
    if "humidity" in df.columns:
        df["grip_index"] = (1 - df["humidity"]/100.0) * (1 - df["rain_factor"])
    else:
        df["grip_index"] = 0.5
    if "wind_speed" in df.columns:
        df["wind_penalty"] = np.clip(df["wind_speed"]/60.0, 0, 1)
    else:
        df["wind_penalty"] = 0.1
    df["weather_score"] = (
        0.4 * df["grip_index"] +
        0.3 * (1 - df["wind_penalty"]) +
        0.3 * np.clip(df["temp_effect"], 0.4, 0.9)
    )
    return df


# ── Performance Rolling Features ─────────────────────────────────────

def add_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["year", "round"])

    for driver, grp in df.groupby("driver_id"):
        idx = grp.index
        df.loc[idx, "driver_avg_pos_5"] = (
            grp["position"].shift(1).rolling(5, min_periods=1).mean()
        )
        df.loc[idx, "driver_avg_pts_5"] = (
            grp["points"].shift(1).rolling(5, min_periods=1).mean()
        )
        df.loc[idx, "driver_wins_5"] = (
            (grp["position"].shift(1) == 1).rolling(5, min_periods=1).sum()
        )
        df.loc[idx, "driver_podiums_5"] = (
            (grp["position"].shift(1) <= 3).rolling(5, min_periods=1).sum()
        )
        df.loc[idx, "driver_dnf_rate"] = (
            grp["is_dnf"].shift(1).rolling(10, min_periods=1).mean()
            if "is_dnf" in df.columns else 0.15
        )
        df.loc[idx, "grid_delta"] = (
            (grp["grid"].shift(1) - grp["position"].shift(1)).rolling(5, min_periods=1).mean()
        )
        pts_shifted = grp["points"].shift(1)
        df.loc[idx, "driver_momentum"] = pts_shifted.ewm(span=5, adjust=False).mean()

    for team, grp in df.groupby("constructor_id"):
        idx = grp.index
        df.loc[idx, "team_avg_pts_5"] = (
            grp["points"].shift(1).rolling(5, min_periods=1).mean()
        )
        df.loc[idx, "team_wins_5"] = (
            (grp["position"].shift(1) == 1).rolling(5, min_periods=1).sum()
        )

    # Circuit-specific driver performance
    if "circuit_id" in df.columns:
        for (driver, circuit), grp in df.groupby(["driver_id", "circuit_id"]):
            idx = grp.index
            df.loc[idx, "driver_circuit_avg"] = (
                grp["position"].shift(1).rolling(3, min_periods=1).mean()
            )

    roll_cols = [
        "driver_avg_pos_5", "driver_avg_pts_5", "driver_wins_5",
        "driver_podiums_5", "driver_dnf_rate", "grid_delta",
        "team_avg_pts_5", "team_wins_5", "driver_momentum", "driver_circuit_avg",
    ]
    for col in roll_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# ── Feature Lists ────────────────────────────────────────────────────

TABULAR_FEATURES = [
    "grid",
    "championship_position", "championship_points",
    "constructor_points", "constructor_position",
    "driver_avg_pos_5", "driver_avg_pts_5",
    "driver_wins_5", "driver_podiums_5",
    "driver_dnf_rate", "grid_delta",
    "team_avg_pts_5", "team_wins_5",
    "driver_momentum", "driver_circuit_avg",
    "rain_factor", "temp_effect", "grip_index",
    "wind_penalty", "weather_score",
    "is_wet", "team_tier",
]

SEQUENCE_FEATURES = [
    "position", "points", "grid",
    "driver_avg_pos_5", "driver_momentum",
    "grip_index", "weather_score",
]

TARGET = "position"


# ── Feature Pipeline ─────────────────────────────────────────────────

class FeaturePipeline:
    def __init__(self, sequence_length: int = 5):
        self.sequence_length = sequence_length
        self.label_encoders: dict = {}
        self.scaler = MinMaxScaler()
        self.tabular_cols: list = []
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame):
        df = self._preprocess(df, fit=True)
        X_tab = df[self.tabular_cols].values.astype(np.float32)
        X_seq = self._build_sequences(df)
        y = df[TARGET].values.astype(np.float32)
        return X_tab, X_seq, y

    def transform(self, df: pd.DataFrame):
        df = self._preprocess(df, fit=False)
        X_tab = df[self.tabular_cols].values.astype(np.float32)
        X_seq = self._build_sequences(df)
        return X_tab, X_seq

    def _preprocess(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        for cat_col in ["driver_id", "constructor_id", "circuit_id", "circuit_type", "nationality"]:
            if cat_col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[cat_col] = le.fit_transform(df[cat_col].fillna("unknown").astype(str))
                self.label_encoders[cat_col] = le
            else:
                le = self.label_encoders.get(cat_col)
                if le:
                    known = set(le.classes_)
                    df[cat_col] = df[cat_col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[cat_col] = le.transform(df[cat_col].fillna(le.classes_[0]))

        self.tabular_cols = [c for c in TABULAR_FEATURES if c in df.columns]
        for cat in ["driver_id", "constructor_id", "circuit_id"]:
            if cat in df.columns and cat not in self.tabular_cols:
                self.tabular_cols.append(cat)

        for col in self.tabular_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if fit:
            df[self.tabular_cols] = self.scaler.fit_transform(df[self.tabular_cols])
        else:
            df[self.tabular_cols] = self.scaler.transform(df[self.tabular_cols])

        self.fitted = True
        return df

    def _build_sequences(self, df: pd.DataFrame) -> np.ndarray:
        seq_cols = [c for c in SEQUENCE_FEATURES if c in df.columns]
        n, seq_len, n_feat = len(df), self.sequence_length, len(seq_cols)
        X_seq = np.zeros((n, seq_len, n_feat), dtype=np.float32)
        driver_col = "driver_id" if "driver_id" in df.columns else None
        if driver_col:
            for driver, grp in df.groupby(driver_col):
                indices = grp.index.tolist()
                vals = grp[seq_cols].values
                for t, pos in enumerate(indices):
                    start = max(0, t - seq_len)
                    chunk = vals[start:t]
                    if len(chunk) == 0:
                        continue
                    pad = seq_len - len(chunk)
                    seq = np.vstack([np.zeros((pad, n_feat)), chunk])
                    row_idx = df.index.get_loc(pos)
                    X_seq[row_idx] = seq
        return X_seq
