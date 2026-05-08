"""
analysis.py
════════════════════════════════════════════════════════════════
Historical analysis of the F1 1950–2026 dataset.
Produces text-based charts and stats tables saved to outputs/.
════════════════════════════════════════════════════════════════
"""

import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.makedirs("outputs", exist_ok=True)


def bar(value, max_val, width=30, char="█"):
    n = int(round(value / max_val * width))
    return char * n + "░" * (width - n)


def print_section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def run_analysis(csv_path="data/f1_1950_2026.csv"):
    print("\n🏎️  F1 HISTORICAL ANALYSIS — 1950 to 2026")

    df = pd.read_csv(csv_path)
    print(f"   Dataset: {len(df):,} rows | {df['year'].nunique()} seasons "
          f"| {df['circuit_id'].nunique()} circuits | {df['driver_id'].nunique()} drivers")

    # ── All-time top drivers (wins) ───────────────────────────
    print_section("All-Time Top Drivers by Race Wins")
    wins = df[df["position"] == 1].groupby("driver_name").size().sort_values(ascending=False).head(20)
    max_w = wins.max()
    for driver, w in wins.items():
        print(f"  {driver:<28} {bar(w, max_w, 25)} {w:4d}")

    # ── Constructor wins ──────────────────────────────────────
    print_section("All-Time Constructor Wins")
    cw = df[df["position"]==1].groupby("constructor_name").size().sort_values(ascending=False).head(15)
    for team, w in cw.items():
        print(f"  {team:<25} {bar(w, cw.max(), 25)} {w:4d}")

    # ── DNF rate by era ───────────────────────────────────────
    print_section("DNF Rate by Era")
    df["era"] = pd.cut(df["year"],
                       bins=[1949,1959,1969,1979,1989,1999,2009,2019,2026],
                       labels=["1950s","1960s","1970s","1980s","1990s","2000s","2010s","2020s"])
    dnf = df.groupby("era", observed=True)["is_dnf"].mean().sort_index()
    for era, rate in dnf.items():
        print(f"  {era}  {bar(rate, dnf.max(), 25)} {rate*100:5.1f}%")

    # ── Wet race % by circuit ─────────────────────────────────
    print_section("Wettest Circuits (% wet races)")
    wet = df.groupby("circuit_name")["is_wet"].mean().sort_values(ascending=False).head(12)
    for circ, pct in wet.items():
        print(f"  {circ[:30]:<32} {bar(pct, wet.max(), 20)} {pct*100:5.1f}%")

    # ── Points eras ───────────────────────────────────────────
    print_section("Max Points per Race by Era")
    era_pts = df.groupby("era", observed=True)["points"].max()
    for era, pts in era_pts.items():
        print(f"  {era}  {bar(pts, 26, 25)} {pts:.0f} pts")

    # ── Circuit longevity ─────────────────────────────────────
    print_section("Most Historic Circuits (years on calendar)")
    years_on = df.groupby("circuit_name")["year"].nunique().sort_values(ascending=False).head(15)
    for c, y in years_on.items():
        print(f"  {c[:30]:<32} {bar(y, years_on.max(), 25)} {y:3d} seasons")

    # ── Season champions summary ──────────────────────────────
    print_section("Season Champions (by most wins per year)")
    champs = []
    for year, grp in df.groupby("year"):
        wins_yr = grp[grp["position"]==1].groupby(["driver_name","constructor_name"]).size()
        if len(wins_yr):
            winner = wins_yr.idxmax()
            champs.append({"year": year, "driver": winner[0], "constructor": winner[1],
                           "wins": wins_yr.max()})
    champ_df = pd.DataFrame(champs)

    # Multi-champions
    mc = champ_df["driver"].value_counts()
    print("\n  Multiple Championship Years:")
    for d, n in mc[mc > 1].items():
        print(f"   🏆 {d:<28} {n}× champion")

    # Save analysis to JSON
    summary = {
        "total_rows":     int(len(df)),
        "seasons":        int(df["year"].nunique()),
        "circuits":       int(df["circuit_id"].nunique()),
        "drivers":        int(df["driver_id"].nunique()),
        "top_drivers":    wins.head(10).to_dict(),
        "top_constructors": cw.head(10).to_dict(),
        "dnf_by_era":     {str(k): round(float(v)*100, 1) for k, v in dnf.items()},
        "champions":      champ_df.tail(20).to_dict(orient="records"),
    }
    with open("outputs/analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  📄 Analysis saved → outputs/analysis.json")


if __name__ == "__main__":
    if not os.path.exists("data/f1_1950_2026.csv"):
        print("Generating dataset first…")
        from data.generate_dataset import generate_full_dataset
        df = generate_full_dataset()
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/f1_1950_2026.csv", index=False)
    run_analysis()
