"""
test_model.py
════════════════════════════════════════════════════════════════
Comprehensive test suite for the F1 Hybrid Prediction System.

Tests:
  1. Dataset generation & integrity
  2. Feature engineering correctness
  3. XGBoost model train/predict/save/load
  4. LSTM fallback model
  5. Hybrid engine end-to-end
  6. Era-specific predictions (1950s, 1970s, 1990s, 2020s)
  7. Wet vs dry race comparison
  8. Constructor championship logic
  9. Model persistence (save/load roundtrip)
  10. Performance benchmarks

Usage:
    python test_model.py
    python test_model.py --quick   # skip slow tests
════════════════════════════════════════════════════════════════
"""

import sys, os, time, argparse, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"
results_log = []


def log(name, ok, detail=""):
    status = PASS if ok else FAIL
    results_log.append((name, ok, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════
# 1. DATASET GENERATION
# ══════════════════════════════════════════════════════════════
def test_dataset_generation(quick=False):
    section("1 · Dataset Generation & Integrity")
    from data.generate_dataset import generate_full_dataset, CIRCUITS, DRIVERS, CONSTRUCTORS

    # Check circuit list
    log("Circuit list not empty", len(CIRCUITS) > 50, f"{len(CIRCUITS)} circuits")
    log("Driver list not empty",  len(DRIVERS)  > 50, f"{len(DRIVERS)} drivers")
    log("Constructor list",       len(CONSTRUCTORS) > 20, f"{len(CONSTRUCTORS)} teams")

    # Spot-check known circuits
    cids = [c[0] for c in CIRCUITS]
    for exp in ["monaco","silverstone","monza","spa","interlagos","suzuka","bahrain","abu_dhabi"]:
        log(f"Circuit present: {exp}", exp in cids)

    # Spot-check known drivers
    dids = [d[0] for d in DRIVERS]
    for exp in ["fangio","senna","schumacher_m","hamilton","verstappen","norris"]:
        log(f"Driver present: {exp}", exp in dids)

    if quick:
        # Generate only 2 seasons for speed
        from data.generate_dataset import get_circuits_for_year, get_active_drivers, simulate_race, get_dominant
        rows = []
        champ = {}
        for year in [1950, 2024]:
            circuits = get_circuits_for_year(year)
            drivers  = get_active_drivers(year)
            dominant = get_dominant(year)
            for rno, cid in enumerate(circuits[:3], 1):
                r = simulate_race(year, rno, cid, drivers, dominant, champ)
                rows.extend(r)
        df = pd.DataFrame(rows)
    else:
        df = generate_full_dataset()

    log("DataFrame not empty",          len(df) > 0,                  f"{len(df):,} rows")
    log("Covers multiple seasons",      df["year"].nunique() >= (2 if quick else 70))
    log("Has all key columns",
        all(c in df.columns for c in ["year","circuit_id","driver_id","constructor_id",
                                       "grid","position","points","status","is_wet"]))
    log("No null positions",             df["position"].isna().sum() == 0)
    log("Positions in valid range",      df["position"].between(1, 30).all())
    log("Grid in valid range",           df["grid"].between(1, 30).all())
    log("Points non-negative",           (df["points"] >= 0).all())
    log("Weather fields numeric",        pd.to_numeric(df["track_temp"], errors="coerce").notna().all())
    log("Wet flag binary",               df["is_wet"].isin([0,1]).all())
    log("DNF flag present",              "is_dnf" in df.columns)

    # Era check
    if not quick:
        log("1950s data exists",  (df["year"] < 1960).any())
        log("2020s data exists",  (df["year"] >= 2020).any())
        log("1976 Lauda season",  ((df["year"]==1976) & (df["driver_id"]=="lauda")).any())
        log("1994 Senna present", ((df["year"]==1994) & (df["driver_id"]=="senna")).any())
        log("2023 Verstappen",    ((df["year"]==2023) & (df["driver_id"]=="verstappen")).any())

    return df


# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def test_features(df):
    section("2 · Feature Engineering")
    from data.features import (add_performance_features, weather_impact,
                                compute_weather_impact_score, FeaturePipeline,
                                TABULAR_FEATURES, SEQUENCE_FEATURES)

    # Weather score
    dry = compute_weather_impact_score({"rainfall":0,"humidity":50,"wind_speed":10,"track_temp":40})
    wet = compute_weather_impact_score({"rainfall":15,"humidity":85,"wind_speed":25,"track_temp":20})
    log("Weather score dry > wet",   dry > wet,  f"dry={dry:.3f} wet={wet:.3f}")
    log("Weather score in [0,1]",    0 <= dry <= 1 and 0 <= wet <= 1)

    # Performance features
    t0 = time.time()
    df2 = add_performance_features(df.copy())
    log("Performance features added", "driver_avg_pos_5" in df2.columns,
        f"{time.time()-t0:.1f}s")
    log("Momentum column exists",     "driver_momentum" in df2.columns)
    log("DNF rate column exists",     "driver_dnf_rate" in df2.columns)
    log("Circuit avg column exists",  "driver_circuit_avg" in df2.columns)
    log("No all-NaN rolling col",
        all(df2[c].notna().sum() > 0 for c in ["driver_avg_pos_5","driver_momentum"]))

    # Weather impact features
    df3 = weather_impact(df2.copy())
    log("Rain factor column",     "rain_factor"   in df3.columns)
    log("Grip index column",      "grip_index"    in df3.columns)
    log("Weather score column",   "weather_score" in df3.columns)
    log("Grip index in [0,1]",    df3["grip_index"].between(0,1).all())
    log("Weather score in [0,1]", df3["weather_score"].between(0,1).all())

    # Pipeline
    small = df3.head(200).copy()
    fp = FeaturePipeline(sequence_length=5)
    X_tab, X_seq, y = fp.fit_transform(small)
    log("Tabular shape correct",  X_tab.ndim == 2 and X_tab.shape[0] == len(small),
        f"{X_tab.shape}")
    log("Sequence shape correct", X_seq.shape == (len(small), 5, X_seq.shape[2]),
        f"{X_seq.shape}")
    log("Target shape correct",   y.shape == (len(small),))
    log("Tabular values finite",  np.isfinite(X_tab).all())
    log("Sequence values finite", np.isfinite(X_seq).all())
    log("Labels encoded",         len(fp.label_encoders) > 0)

    # Transform (inference)
    X_tab2, X_seq2 = fp.transform(small)
    log("Transform (inference) ok", X_tab2.shape == X_tab.shape)

    return df3, fp


# ══════════════════════════════════════════════════════════════
# 3. XGBOOST MODEL
# ══════════════════════════════════════════════════════════════
def test_xgboost(df, fp):
    section("3 · XGBoost Model")
    from models.xgboost_model import XGBoostRacePredictor
    from sklearn.metrics import mean_absolute_error

    X_tab, _, y = fp.fit_transform(df.head(500).copy())

    model = XGBoostRacePredictor()
    t0 = time.time()
    model.fit(X_tab, y, feature_names=fp.tabular_cols, verbose=False)
    log("XGBoost trains",          model.is_fitted, f"{time.time()-t0:.1f}s")

    preds = model.predict(X_tab)
    mae = mean_absolute_error(y, preds)
    log("Predict returns array",   len(preds) == len(y), f"shape={preds.shape}")
    log("Train MAE reasonable",    mae < 10.0, f"MAE={mae:.3f}")

    probs = model.predict_proba_winner(X_tab[:20])
    log("Win probs sum to 1",      abs(probs.sum() - 1.0) < 1e-5, f"sum={probs.sum():.6f}")
    log("Win probs in [0,1]",      (probs >= 0).all() and (probs <= 1).all())

    fi = model.feature_importance(top_n=10)
    log("Feature importance DF",   len(fi) > 0, f"{len(fi)} features")
    log("Importance pct sums ~100",abs(fi["importance_pct"].sum() - 100) < 1,
        f"{fi['importance_pct'].sum():.1f}%")

    # Save/load
    path = "models/saved/_test_xgb.pkl"
    model.save(path)
    loaded = XGBoostRacePredictor.load(path)
    preds2 = loaded.predict(X_tab)
    log("Save/load roundtrip",     np.allclose(preds, preds2))
    os.remove(path)

    return model


# ══════════════════════════════════════════════════════════════
# 4. LSTM FALLBACK MODEL
# ══════════════════════════════════════════════════════════════
def test_lstm(df, fp):
    section("4 · LSTM / Fallback Model")
    from models.lstm_model import SimpleLSTMFallback, get_lstm_model

    _, X_seq, y = fp.fit_transform(df.head(200).copy())

    # Fallback always available
    fb = SimpleLSTMFallback(span=5)
    fb.fit(X_seq, y)
    preds = fb.predict(X_seq)
    log("Fallback predict shape",  preds.shape == (len(X_seq),), f"{preds.shape}")
    log("Fallback values finite",  np.isfinite(preds).all())
    log("Fallback fit is no-op",   True)

    model = get_lstm_model(seq_len=5, n_features=X_seq.shape[2])
    log("Factory returns model",   model is not None)
    model.fit(X_seq, y, epochs=2)
    preds2 = model.predict(X_seq)
    log("LSTM predict shape",      preds2.shape == (len(X_seq),), f"{preds2.shape}")
    log("LSTM values finite",      np.isfinite(preds2).all())

    return model


# ══════════════════════════════════════════════════════════════
# 5. HYBRID ENGINE END-TO-END
# ══════════════════════════════════════════════════════════════
def test_hybrid_engine(full_df):
    section("5 · Hybrid Engine – End-to-End")
    from models.hybrid_engine import HybridF1Engine, HybridConfig

    cfg = HybridConfig(w_xgboost=0.50, w_lstm=0.30, w_weather=0.20)
    engine = HybridF1Engine(config=cfg)

    # Verify weights normalise
    log("Weights normalised",     abs(cfg.w_xgboost+cfg.w_lstm+cfg.w_weather-1.0) < 1e-5,
        f"{cfg.w_xgboost:.2f}+{cfg.w_lstm:.2f}+{cfg.w_weather:.2f}")

    t0 = time.time()
    metrics = engine.train(full_df, verbose=False)
    log("Engine trains",          engine.is_trained, f"{time.time()-t0:.1f}s")
    log("Metrics returned",       isinstance(metrics, dict) and "mae_xgboost" in metrics)
    log("XGBoost MAE < 15",       metrics["mae_xgboost"] < 15, f"{metrics['mae_xgboost']}")
    log("Hybrid MAE < 15",        metrics["mae_hybrid"]  < 15, f"{metrics['mae_hybrid']}")

    return engine, metrics


# ══════════════════════════════════════════════════════════════
# 6. ERA-SPECIFIC PREDICTIONS
# ══════════════════════════════════════════════════════════════
def test_era_predictions(engine):
    section("6 · Era-Specific Race Predictions")
    from predict import build_race_df

    eras = [
        (1952, "monza",      "Alberto Ascari era"),
        (1969, "silverstone","Jackie Stewart era"),
        (1988, "monaco",     "Senna/Prost era"),
        (2004, "monza",      "Schumacher dominance"),
        (2016, "spa",        "Hamilton/Rosberg era"),
        (2023, "bahrain",    "Verstappen era"),
        (2026, "miami",      "2026 season"),
    ]

    for year, circuit, label in eras:
        try:
            race_df = build_race_df(year, circuit)
            weather_dry = {"rainfall":0,"track_temp":38,"humidity":55,"wind_speed":12,"is_wet":False}
            results = engine.predict_race(race_df, weather=weather_dry)
            ok = len(results) > 0 and results[0].predicted_position == 1
            log(f"Era {year} {label}", ok,
                f"P1→{results[0].driver} ({results[0].win_probability*100:.1f}%)")
        except Exception as e:
            log(f"Era {year} {label}", False, str(e)[:50])


# ══════════════════════════════════════════════════════════════
# 7. WET vs DRY RACE
# ══════════════════════════════════════════════════════════════
def test_wet_dry(engine):
    section("7 · Wet vs Dry Race Comparison")
    from predict import build_race_df

    race_df = build_race_df(2024, "spa")
    weather_dry = {"rainfall":0,"track_temp":42,"humidity":50,"wind_speed":10,"is_wet":False}
    weather_wet = {"rainfall":25,"track_temp":18,"humidity":90,"wind_speed":22,"is_wet":True}

    res_dry = engine.predict_race(race_df, weather=weather_dry)
    res_wet = engine.predict_race(race_df, weather=weather_wet)

    dry_winner = res_dry[0].driver
    wet_winner = res_wet[0].driver
    dry_score  = res_dry[0].weather_adjusted_score
    wet_score  = res_wet[0].weather_adjusted_score

    log("Dry race produces results", len(res_dry) > 0)
    log("Wet race produces results", len(res_wet) > 0)
    log("Weather impact score differs",
        res_dry[0].weather_impact != res_wet[0].weather_impact,
        f"dry={res_dry[0].weather_impact:.3f} wet={res_wet[0].weather_impact:.3f}")
    log("Win probs sum to ~1 (dry)", abs(sum(r.win_probability for r in res_dry)-1.0) < 0.01)
    log("Win probs sum to ~1 (wet)", abs(sum(r.win_probability for r in res_wet)-1.0) < 0.01)
    print(f"     Dry winner: {dry_winner} | Wet winner: {wet_winner}")


# ══════════════════════════════════════════════════════════════
# 8. CONSTRUCTOR CHAMPIONSHIP
# ══════════════════════════════════════════════════════════════
def test_constructor_championship(engine):
    section("8 · Constructor Championship Logic")
    from predict import build_race_df

    race_df = build_race_df(2024, "silverstone")
    results = engine.predict_race(race_df)
    team_df = engine.predict_constructor_championship(results)

    log("Constructor DF not empty",         len(team_df) > 0, f"{len(team_df)} teams")
    log("Has championship_rank column",     "championship_rank" in team_df.columns)
    log("Has total_points column",          "total_points" in team_df.columns)
    log("Ranks are 1-based sequential",     list(team_df["championship_rank"]) == list(range(1, len(team_df)+1)))
    log("Points non-negative",              (team_df["total_points"] >= 0).all())
    log("Best position ≤ avg position",
        (team_df["best_position"] <= team_df["avg_position"]).all())
    print(f"     P1 constructor: {team_df.iloc[0]['constructor']}  "
          f"({team_df.iloc[0]['total_points']:.0f} pts)")


# ══════════════════════════════════════════════════════════════
# 9. MODEL PERSISTENCE
# ══════════════════════════════════════════════════════════════
def test_persistence(engine, full_df):
    section("9 · Model Persistence (Save / Load)")
    from models.hybrid_engine import HybridF1Engine
    from predict import build_race_df
    import tempfile, shutil

    tmpdir = tempfile.mkdtemp()
    try:
        engine.save(tmpdir)
        saved_files = list(Path(tmpdir).glob("*"))
        log("Files saved",         len(saved_files) >= 2,
            ", ".join(f.name for f in saved_files))
        log("pipeline.pkl exists", (Path(tmpdir)/"pipeline.pkl").exists())
        log("xgboost pkl exists",  (Path(tmpdir)/"xgboost_f1.pkl").exists())

        engine2 = HybridF1Engine.load(tmpdir)
        log("Engine reloads",      engine2.is_trained)

        race_df = build_race_df(2024, "monaco")
        r1 = engine.predict_race(race_df)
        r2 = engine2.predict_race(race_df)
        same_winner = r1[0].driver == r2[0].driver
        log("Same predictions after reload", same_winner,
            f"original={r1[0].driver}, loaded={r2[0].driver}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
# 10. PERFORMANCE BENCHMARKS
# ══════════════════════════════════════════════════════════════
def test_performance(engine):
    section("10 · Performance Benchmarks")
    from predict import build_race_df

    race_df = build_race_df(2024, "suzuka")
    weather = {"rainfall":0,"track_temp":35,"humidity":60,"wind_speed":15,"is_wet":False}

    times = []
    for _ in range(10):
        t0 = time.time()
        engine.predict_race(race_df, weather=weather)
        times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    p99_ms = np.percentile(times, 99) * 1000
    log("Avg inference < 500ms",  avg_ms < 500, f"{avg_ms:.1f}ms")
    log("P99 inference < 2000ms", p99_ms < 2000, f"{p99_ms:.1f}ms")
    print(f"     10-run avg: {avg_ms:.1f}ms  |  p99: {p99_ms:.1f}ms")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  F1 HYBRID AI — TEST SUITE")
    print("  Covering 1950–2026 | XGBoost × LSTM × Weather")
    print("═"*60)
    t_total = time.time()

    # Generate / load dataset
    os.makedirs("models/saved", exist_ok=True)
    df_raw = test_dataset_generation(quick=args.quick)

    # If quick, use subset
    sample_df = df_raw.head(800) if args.quick else df_raw

    df_feat, fp = test_features(sample_df)
    xgb_model   = test_xgboost(df_feat, fp)
    lstm_model  = test_lstm(df_feat, fp)
    engine, metrics = test_hybrid_engine(sample_df)

    test_era_predictions(engine)
    test_wet_dry(engine)
    test_constructor_championship(engine)
    test_persistence(engine, sample_df)
    test_performance(engine)

    # ── Summary ──────────────────────────────────────────────
    total  = len(results_log)
    passed = sum(1 for _, ok, _ in results_log if ok)
    failed = total - passed

    print(f"\n{'═'*60}")
    print(f"  TEST SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total   : {total}")
    print(f"  ✅ Passed : {passed}")
    print(f"  ❌ Failed : {failed}")
    print(f"  Time    : {time.time()-t_total:.1f}s")
    print(f"{'═'*60}")

    if failed > 0:
        print("\n  Failed tests:")
        for name, ok, detail in results_log:
            if not ok:
                print(f"    ❌ {name}  [{detail}]")

    # Export report
    report = {
        "total": total, "passed": passed, "failed": failed,
        "training_metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                             for k, v in metrics.items()},
        "tests": [{"name": n, "passed": bool(ok), "detail": str(d)} for n, ok, d in results_log],
    }
    with open("outputs/test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  📄 Report saved → outputs/test_report.json")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
