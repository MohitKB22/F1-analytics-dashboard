"""
train.py
════════════════════════════════════════════════════════════════
Train the F1 Hybrid Prediction System on 1950-2026 dataset.

Usage:
    python train.py                        # auto-generates dataset
    python train.py --csv data/f1_1950_2026.csv   # use existing CSV
    python train.py --years 1990 2026      # custom year range
    python train.py --cv                   # run cross-validation too
════════════════════════════════════════════════════════════════
"""

import argparse
import sys
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Train F1 Hybrid AI (1950-2026)")
    p.add_argument("--csv",       default="data/f1_1950_2026.csv")
    p.add_argument("--years",     nargs=2, type=int, default=None,
                   help="Year range, e.g. --years 1990 2026")
    p.add_argument("--model-dir", default="models/saved")
    p.add_argument("--w-xgb",     type=float, default=0.50)
    p.add_argument("--w-lstm",    type=float, default=0.30)
    p.add_argument("--w-weather", type=float, default=0.20)
    p.add_argument("--cv",        action="store_true", help="Run 5-fold CV")
    p.add_argument("--regen",     action="store_true", help="Regenerate dataset")
    return p.parse_args()


def load_or_generate(args) -> pd.DataFrame:
    if not args.regen and os.path.exists(args.csv):
        print(f"[Data] Loading {args.csv}…")
        df = pd.read_csv(args.csv)
    else:
        print("[Data] Generating 1950–2026 dataset (this takes ~30s)…")
        from data.generate_dataset import generate_full_dataset
        df = generate_full_dataset()
        os.makedirs("data", exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"[Data] Saved → {args.csv}")

    if args.years:
        y0, y1 = args.years
        df = df[(df["year"] >= y0) & (df["year"] <= y1)]
        print(f"[Data] Filtered to {y0}–{y1}: {len(df):,} rows")

    return df


def main():
    args = parse_args()
    df = load_or_generate(args)

    print(f"\n[Data] {len(df):,} rows | {df['driver_id'].nunique()} drivers | "
          f"{df['circuit_id'].nunique()} circuits | {df['year'].nunique()} seasons")

    from models.hybrid_engine import HybridF1Engine, HybridConfig
    cfg = HybridConfig(
        w_xgboost=args.w_xgb, w_lstm=args.w_lstm, w_weather=args.w_weather,
        model_dir=args.model_dir
    )
    engine = HybridF1Engine(config=cfg)
    metrics = engine.train(df, verbose=True)

    # ── Hold-out evaluation (last 2 seasons) ────────────────
    print("\n[Eval] Hold-out test: 2025-2026 seasons…")
    from data.features import add_performance_features, weather_impact, FeaturePipeline
    test_df = df[df["year"] >= 2025].copy()
    train_df = df[df["year"] < 2025].copy()

    if len(test_df) > 0 and len(train_df) > 0:
        try:
            fp = engine.feature_pipeline
            test_df = add_performance_features(test_df)
            test_df = weather_impact(test_df)
            X_tab_test, X_seq_test = fp.transform(test_df)
            y_test = test_df["position"].values

            xgb_preds  = engine.xgb_model.predict(X_tab_test)
            lstm_preds = engine.lstm_model.predict(X_seq_test)
            hybrid_preds = cfg.w_xgboost * xgb_preds + cfg.w_lstm * lstm_preds

            print(f"   XGBoost  MAE (holdout) : {mean_absolute_error(y_test, xgb_preds):.3f}")
            print(f"   LSTM     MAE (holdout) : {mean_absolute_error(y_test, lstm_preds):.3f}")
            print(f"   Hybrid   MAE (holdout) : {mean_absolute_error(y_test, hybrid_preds):.3f}")
        except Exception as e:
            print(f"   [Eval skipped: {e}]")

    # ── Optional CV ──────────────────────────────────────────
    if args.cv:
        print("\n[CV] Running 5-fold cross-validation on XGBoost…")
        from data.features import add_performance_features, weather_impact
        cv_df = add_performance_features(df.copy())
        cv_df = weather_impact(cv_df)
        X_tab, _, y = engine.feature_pipeline.fit_transform(cv_df)
        cv_result = engine.xgb_model.cross_validate(X_tab, y, cv=5)
        print(f"   CV MAE: {cv_result['cv_mae_mean']:.3f} ± {cv_result['cv_mae_std']:.3f}")

    # ── Feature importance ───────────────────────────────────
    print("\n[Features] Top-15 XGBoost feature importances:")
    fi = engine.xgb_model.feature_importance(top_n=15)
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance_pct"] / 2)
        print(f"   {row['feature']:<30} {bar}  {row['importance_pct']:.1f}%")

    # ── Save ─────────────────────────────────────────────────
    engine.save(args.model_dir)

    print(f"\n✅  Training complete!")
    print(f"   XGBoost MAE  : {metrics['mae_xgboost']}")
    print(f"   LSTM    MAE  : {metrics['mae_lstm']}")
    print(f"   Hybrid  MAE  : {metrics['mae_hybrid']}")
    print(f"   Samples      : {metrics['n_samples']:,}")
    print(f"   Tab features : {metrics['n_tab_features']}")
    print(f"   Models saved → {args.model_dir}/")


if __name__ == "__main__":
    main()
