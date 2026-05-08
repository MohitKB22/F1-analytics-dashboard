#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# run.sh  —  F1 Hybrid AI  •  Mac / Linux setup & launch helper
#
# Usage:
#   chmod +x run.sh
#   ./run.sh              # full setup + launch Streamlit
#   ./run.sh train        # setup + train model only
#   ./run.sh predict      # setup + run a sample prediction
#   ./run.sh test         # setup + run test suite
# ═══════════════════════════════════════════════════════════════

set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ── Colours ────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
CYN='\033[0;36m'; WHT='\033[1;37m'; NC='\033[0m'

banner() { echo -e "\n${WHT}════════════════════════════════════════${NC}"; \
           echo -e "${RED}  🏎️  F1 HYBRID AI — $1${NC}"; \
           echo -e "${WHT}════════════════════════════════════════${NC}\n"; }

step()   { echo -e "${CYN}▶ $1${NC}"; }
ok()     { echo -e "${GRN}✅ $1${NC}"; }
warn()   { echo -e "${YLW}⚠️  $1${NC}"; }
fail()   { echo -e "${RED}❌ $1${NC}"; exit 1; }

banner "Setup & Launch"

# ── 1. Python version check ─────────────────────────────────────
step "Checking Python version…"
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
        # Accept 3.9 – 3.13
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)")
        MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ] && [ "$MINOR" -le 13 ]; then
            PYTHON="$cmd"
            ok "Using $cmd (Python $MAJOR.$MINOR)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python 3.9–3.13 not found.\n\n  Install via:\n    brew install python@3.12    # Mac\n    sudo apt install python3.12  # Ubuntu\n\n  Then re-run this script."
fi

# ── 2. Mac OpenMP fix (needed by XGBoost) ──────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    step "Mac detected — checking OpenMP (required by XGBoost)…"
    if ! brew list libomp &>/dev/null 2>&1; then
        warn "libomp not found. Installing via Homebrew…"
        if ! command -v brew &>/dev/null; then
            fail "Homebrew not found. Install it first:\n  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"\nthen re-run this script."
        fi
        brew install libomp
        ok "libomp installed"
    else
        ok "libomp already installed"
    fi
fi

# ── 3. Virtual environment ─────────────────────────────────────
step "Setting up virtual environment…"
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
    ok "Created .venv"
else
    ok ".venv already exists"
fi

# Activate
source .venv/bin/activate
PYTHON=python   # use venv python from here on

# ── 4. Install dependencies ─────────────────────────────────────
step "Installing dependencies (this takes ~60s first time)…"
pip install --upgrade pip -q
pip install -e . -q
ok "Dependencies installed"

# ── 5. Verify XGBoost loads ────────────────────────────────────
step "Verifying XGBoost…"
if ! python -c "import xgboost" 2>/dev/null; then
    warn "XGBoost import failed. On Mac, try:\n  brew install libomp\nthen re-run."
    exit 1
fi
ok "XGBoost OK"

# ── 6. Train if no saved model ─────────────────────────────────
if [ ! -f "models/saved/xgboost_f1.pkl" ] || [ ! -f "models/saved/pipeline.pkl" ]; then
    step "No trained model found — training now (takes ~30s)…"
    python train.py
    ok "Model trained and saved"
else
    ok "Trained model already exists — skipping training"
fi

# ── 7. Run requested action ────────────────────────────────────
ACTION="${1:-streamlit}"

case "$ACTION" in
  train)
    banner "Training"
    python train.py
    ;;
  predict)
    banner "Sample Predictions"
    echo -e "\n${YLW}Monaco 2024 (Dry):${NC}"
    python predict.py --circuit monaco --year 2024
    echo -e "\n${YLW}Spa 1988 (Wet — Senna/Prost era):${NC}"
    python predict.py --circuit spa --year 1988 --rainfall 20
    echo -e "\n${YLW}Silverstone 1969 (Clark era):${NC}"
    python predict.py --circuit silverstone --year 1969
    ;;
  test)
    banner "Test Suite"
    python test_model.py
    ;;
  analysis)
    banner "Historical Analysis"
    python analysis.py
    ;;
  streamlit|"")
    banner "Launching Streamlit Dashboard"
    echo -e "${GRN}  Dashboard will open at: http://localhost:8501${NC}\n"
    streamlit run app.py
    ;;
  *)
    echo "Unknown action: $ACTION"
    echo "Usage: ./run.sh [streamlit|train|predict|test|analysis]"
    exit 1
    ;;
esac
