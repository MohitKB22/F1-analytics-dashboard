# RaceIQ-F1 — Hybrid AI Formula 1 Prediction System

> **XGBoost × LSTM × Weather Intelligence**
> Race prediction across 77 seasons, 67 real circuits, and 74 drivers — 1950 to 2026.

---

## ⚡ Quickest Start (one command)

```bash
# Mac / Linux
chmod +x run.sh && ./run.sh

# Windows (Command Prompt or double-click)
run.bat
```

This script handles **everything**: Python version check, OpenMP fix (Mac), virtual environment, install, training, and Streamlit launch.

---

## 🔮 CLI Predictions

```bash
# 2024 Monaco — dry
python predict.py --circuit monaco --year 2024

# 1988 Spa — heavy rain (Senna/Prost era)
python predict.py --circuit spa --year 1988 --rainfall 20 --track-temp 16

# 1967 Silverstone (Jim Clark era)
python predict.py --circuit silverstone --year 1967

# List all 67 circuits
python predict.py --list-circuits
```

---

## 📋 Command Reference

| Command | What it does |
|---------|-------------|
| `./run.sh` | Full setup + Streamlit (Mac/Linux) |
| `run.bat` | Full setup + Streamlit (Windows) |
| `python train.py` | Train / retrain the model |
| `python predict.py --circuit monaco --year 2024` | Predict a race |
| `python predict.py --list-circuits` | List all 67 circuits |
| `streamlit run app.py` | Launch dashboard |
| `python test_model.py` | Run 90-test suite |
| `python analysis.py` | Historical stats |

---

## 🏁 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | All-time wins, constructor trends, DNF rate by decade |
| 🔮 Race Predictor | Any circuit/year/weather → podium + full grid + win % |
| 📊 Driver Analytics | Multi-driver wins, points, DNF rate, scatter plots |
| 🏆 Championships | 77 season champions, multi-title table, constructor titles |
| 🌧 Weather Lab | Dry vs wet probability comparison simulator |
| 🗺 Circuit Map | Interactive world map of all 67 venues |

---
  <sub>Built with XGBoost · LSTM · Streamlit · Plotly &nbsp;|&nbsp; 1950 – 2026</sub>
</p>
