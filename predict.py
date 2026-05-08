"""
predict.py
════════════════════════════════════════════════════════════════
Run race predictions for any circuit and year (1950-2026).

Usage:
    python predict.py --circuit monaco --year 2024
    python predict.py --circuit spa --year 1988 --rainfall 20
    python predict.py --circuit silverstone --year 1967
    python predict.py --list-circuits
════════════════════════════════════════════════════════════════
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.generate_dataset import CIRCUITS, DRIVERS, CONSTRUCTORS, get_dominant
from data.features import compute_weather_impact_score


DRIVER_COLORS = {
    "red_bull":    "🔵", "ferrari":     "🔴", "mclaren":     "🟠",
    "mercedes":    "⚪", "aston_martin":"🟢", "alpine":      "💙",
    "williams":    "🩵", "haas":        "⬛", "sauber":      "🟡",
    "lotus":       "🟡", "tyrrell":     "🩵", "brabham":     "⬜",
    "maserati":    "🔴", "alfa_romeo_50":"🔴","vanwall":     "🟢",
}


def parse_args():
    p = argparse.ArgumentParser(description="F1 Hybrid Race Predictor (1950-2026)")
    p.add_argument("--circuit",    default="monaco")
    p.add_argument("--year",       type=int, default=2024)
    p.add_argument("--rainfall",   type=float, default=0.0)
    p.add_argument("--track-temp", type=float, default=38.0)
    p.add_argument("--humidity",   type=float, default=55.0)
    p.add_argument("--wind",       type=float, default=12.0)
    p.add_argument("--model-dir",  default="models/saved")
    p.add_argument("--list-circuits", action="store_true")
    p.add_argument("--top",        type=int, default=20)
    return p.parse_args()


def list_circuits():
    print(f"\n{'Circuit ID':<25} {'Name':<35} {'Country':<15} {'Years'}")
    print("─" * 90)
    for c in sorted(CIRCUITS, key=lambda x: x[4]):
        print(f"{c[0]:<25} {c[1]:<35} {c[2]:<15} {c[4]}–{c[5]}")


def build_race_df(year: int, circuit_id: str) -> pd.DataFrame:
    active = [d for d in DRIVERS if d[3] <= year <= d[4]]
    if not active:
        print(f"No drivers found for year {year}.")
        sys.exit(1)

    dominant = get_dominant(year)
    rows = []
    for d in active[:20]:
        did, dname, nat, y0, y1, skill, main_team = d
        team_options = [c for c in CONSTRUCTORS if c[2] <= year <= c[3]]
        if any(t[0] == main_team and t[2] <= year <= t[3] for t in CONSTRUCTORS):
            team_id = main_team
        else:
            team_id = team_options[0][0] if team_options else "unknown"
        team_name = next((t[1] for t in CONSTRUCTORS if t[0] == team_id), "Unknown")
        team_tier = next((t[4] for t in CONSTRUCTORS if t[0] == team_id), 3)

        elapsed   = max(0, year - y0)
        career_len = y1 - y0 + 1
        age_factor = 1.0 - 0.002 * max(0, elapsed - career_len * 0.6)
        cur_skill  = min(skill * age_factor, 0.99)
        bonus      = 0.12 if team_id == dominant else 0
        grid_score = cur_skill + bonus + np.random.normal(0, 0.06)

        rows.append({
            "driver_id": did, "driver_name": dname, "nationality": nat,
            "constructor_id": team_id, "constructor_name": team_name,
            "team_tier": team_tier,
            "grid": 0,   # filled after sort
            "championship_position": np.random.randint(1, len(active)+1),
            "championship_points": max(0, cur_skill * 300 + np.random.normal(0, 30)),
            "constructor_points": max(0, (1 - team_tier/5) * 400),
            "constructor_position": team_tier,
            "driver_avg_pos_5":  max(1, (1 - cur_skill) * 15 + np.random.normal(0, 1.5)),
            "driver_avg_pts_5":  max(0, cur_skill * 20 + np.random.normal(0, 2)),
            "driver_wins_5":     max(0, round(cur_skill * 3)),
            "driver_podiums_5":  max(0, round(cur_skill * 4)),
            "driver_dnf_rate":   max(0, 0.15 - cur_skill * 0.08 + np.random.uniform(0, 0.05)),
            "grid_delta":        np.random.uniform(-1, 2),
            "team_avg_pts_5":    max(0, (1 - team_tier/5) * 25),
            "team_wins_5":       max(0, round((1 - team_tier/5) * 2)),
            "driver_circuit_avg": max(1, (1 - cur_skill) * 12 + np.random.normal(0, 2)),
            "driver_momentum":   max(0, cur_skill * 18),
            "position": int((1 - cur_skill) * 15 + np.random.randint(1, 6)),
            "points":   max(0, cur_skill * 20),
            "is_wet":   0, "rain_factor": 0,
            "year": year, "round": 1, "circuit_id": circuit_id,
            "_grid_score": grid_score,
        })

    df = pd.DataFrame(rows).sort_values("_grid_score", ascending=False).reset_index(drop=True)
    df["grid"] = range(1, len(df)+1)
    df = df.drop(columns=["_grid_score"])
    return df


def print_results(results, weather, year, circuit_id, top=20):
    w_score = compute_weather_impact_score(weather)
    cond    = weather.get("weather_condition", "Dry")
    rain    = weather.get("rainfall", 0)

    circuit_info = next((c for c in CIRCUITS if c[0] == circuit_id), None)
    circuit_name = circuit_info[1] if circuit_info else circuit_id
    circuit_cntry = circuit_info[2] if circuit_info else ""

    print(f"\n{'═'*65}")
    print(f"  🏎️  F1 HYBRID AI — {circuit_name.upper()}")
    print(f"  📅  Season {year}  |  {circuit_cntry}")
    print(f"  🌤  {cond}  |  Rain: {rain:.1f}mm  |  Weather Score: {w_score:.3f}")
    print(f"{'═'*65}")
    print(f"  {'P':<4} {'Driver':<24} {'Constructor':<20} {'Win%':>6}  Score")
    print(f"{'─'*65}")

    medals = {1:"🥇 P1", 2:"🥈 P2", 3:"🥉 P3"}
    for r in results[:top]:
        medal = medals.get(r.predicted_position, f"   P{r.predicted_position:2d}")
        icon  = DRIVER_COLORS.get(r.constructor.lower().replace(" ","_"), "⬜")
        print(f"  {medal}  {r.driver:<24} {icon}{r.constructor:<18} "
              f"{r.win_probability*100:5.1f}%  {r.weather_adjusted_score:.3f}")

    print(f"{'═'*65}")
    print(f"\n🏆  Predicted Winner: {results[0].driver}  "
          f"({results[0].constructor}) — {results[0].win_probability*100:.1f}%")


def main():
    args = parse_args()

    if args.list_circuits:
        list_circuits()
        return

    weather = {
        "air_temp":    args.track_temp * 0.72,
        "track_temp":  args.track_temp,
        "humidity":    args.humidity,
        "rainfall":    args.rainfall,
        "wind_speed":  args.wind,
        "weather_condition": "Wet" if args.rainfall > 0 else "Dry",
        "is_wet":      args.rainfall > 0,
    }

    race_df = build_race_df(args.year, args.circuit)

    model_dir = args.model_dir
    if os.path.exists(f"{model_dir}/pipeline.pkl"):
        from models.hybrid_engine import HybridF1Engine
        engine = HybridF1Engine.load(model_dir)
    else:
        print("[!] No trained model found. Run train.py first.")
        sys.exit(1)

    results = engine.predict_race(race_df, weather=weather)
    print_results(results, weather, args.year, args.circuit, top=args.top)

    team_df = engine.predict_constructor_championship(results)
    print("\n🏁  Constructor Championship Forecast:")
    print(team_df[["championship_rank","constructor","total_points",
                   "avg_position","best_position"]].to_string(index=False))


if __name__ == "__main__":
    main()
