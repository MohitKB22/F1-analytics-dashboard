# 🏎️ F1 Analytics Dashboard

> **A Hybrid AI Formula 1 Race Prediction & Analytics Platform**
> Powered by XGBoost × LSTM × Weather Intelligence — spanning 77 seasons, 67 circuits, and 74 drivers from **1950 to 2026**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 About

**F1 Analytics Dashboard** is an end-to-end machine learning system that predicts Formula 1 race outcomes and surfaces deep historical analytics through an interactive web dashboard.

At its core, the system blends two complementary models:

- **XGBoost** — a gradient-boosted tree model trained on driver performance, constructor strength, circuit characteristics, qualifying results, and historical head-to-head records.
- **LSTM (Long Short-Term Memory)** — a recurrent neural network that captures sequential momentum patterns: driver form across recent races, championship-pressure effects, and season-arc trajectories.
- **Weather Intelligence** — a wet/dry condition simulator that re-runs predictions under rain or dry scenarios, exposing how sensitive each driver's win probability is to track conditions.

Together, these layers produce full-grid predictions with podium probabilities and win percentages for any circuit, year, and weather condition in the dataset.

---

## ⚡ Quickstart (One Command)

```bash
# macOS / Linux
chmod +x run.sh && ./run.sh

# Windows (Command Prompt or double-click)
run.bat
```

The script handles **everything** automatically:
- Python version verification
- OpenMP compatibility fix (macOS)
- Virtual environment creation
- Dependency installation
- Model training
- Streamlit dashboard launch

---

## 🏁 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | All-time wins leaderboard, constructor trend lines, DNF rate by decade |
| 🔮 **Race Predictor** | Select any circuit, year, and weather → get full grid prediction with win % |
| 📊 **Driver Analytics** | Multi-driver comparison: wins, points, DNF rate, performance scatter plots |
| 🏆 **Championships** | 77 season champions, multi-title holders table, constructor title history |
| 🌧️ **Weather Lab** | Side-by-side dry vs. wet probability simulator for any race scenario |
| 🗺️ **Circuit Map** | Interactive world map of all 67 F1 venues with circuit metadata |

---

## 🧠 How the Prediction Works

```
Race Input (Circuit + Year + Weather)
          │
          ▼
  Feature Engineering
  ┌──────────────────────────────────────┐
  │ • Driver career stats & recent form  │
  │ • Constructor performance index      │
  │ • Circuit-specific historical win %  │
  │ • Grid position & qualifying delta   │
  │ • Weather condition encoding         │
  └──────────────────────────────────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
 XGBoost      LSTM
 (static)   (sequential)
    └─────┬──────┘
          ▼
   Ensemble Output
   → Full grid ranking
   → Podium probabilities
   → Win % per driver
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Plotly |
| ML Models | XGBoost, Keras/TensorFlow (LSTM) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Mapping | Plotly Geo / Mapbox |
| Dataset | 1950–2026 F1 historical data (77 seasons) |

---

## 📊 Dataset Coverage

| Attribute | Coverage |
|-----------|----------|
| Seasons | 1950 – 2026 (77 seasons) |
| Circuits | 67 unique venues worldwide |
| Drivers | 74 drivers across all eras |
| Features | Qualifying times, grid positions, lap counts, DNF reasons, constructor points, weather |

---

<p align="center">
  <sub>Built with XGBoost · LSTM · Streamlit · Plotly &nbsp;|&nbsp; 1950 – 2026 &nbsp;|&nbsp; by <a href="https://github.com/MohitKB22">MohitKB22</a></sub>
</p>
