#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  CineML — Movie Recommendation System
#  One-command setup & launch for macOS
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
#
#  What this does:
#    1. Checks for Python 3.8+
#    2. Creates a virtual environment
#    3. Installs all dependencies
#    4. Copies .env.example → .env (prompts for TMDB key)
#    5. Trains the ML model (builds similarity matrix)
#    6. Launches Flask on http://localhost:5000
# ═══════════════════════════════════════════════════════════════════

set -e  # exit on any error

# ── Colours ───────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

print_banner() {
cat << 'EOF'

  ██████╗██╗███╗   ██╗███████╗███╗   ███╗██╗
 ██╔════╝██║████╗  ██║██╔════╝████╗ ████║██║
 ██║     ██║██╔██╗ ██║█████╗  ██╔████╔██║██║
 ██║     ██║██║╚██╗██║██╔══╝  ██║╚██╔╝██║██║
 ╚██████╗██║██║ ╚████║███████╗██║ ╚═╝ ██║███████╗
  ╚═════╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝╚══════╝
  Movie Recommendation Engine — ML + Flask + TMDB
EOF
}

info()    { echo -e "${CYAN}▶ $1${RESET}"; }
success() { echo -e "${GREEN}✓ $1${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $1${RESET}"; }
error()   { echo -e "${RED}✗ $1${RESET}"; exit 1; }
section() { echo -e "\n${BOLD}━━━  $1  ━━━${RESET}"; }

print_banner

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Step 1: Python ────────────────────────────────────────────────
section "Step 1 — Python"

PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    MAJOR=${VER%%.*}; MINOR=${VER##*.}
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 8 ]; then
      PYTHON="$cmd"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  warn "Python 3.8+ not found."
  if command -v brew &>/dev/null; then
    info "Installing Python via Homebrew …"
    brew install python@3.11
    PYTHON=python3.11
  else
    error "Please install Python 3.8+ from https://python.org and re-run."
  fi
fi

PY_VERSION=$("$PYTHON" --version 2>&1)
success "Using $PY_VERSION at $(which $PYTHON)"

# ── Step 2: Virtual environment ───────────────────────────────────
section "Step 2 — Virtual Environment"

VENV_DIR="$SCRIPT_DIR/venv"
if [ -d "$VENV_DIR" ]; then
  info "Virtual environment already exists."
else
  info "Creating virtual environment …"
  "$PYTHON" -m venv "$VENV_DIR"
  success "venv created at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
success "Activated: $(which python)"

# ── Step 3: Install dependencies ──────────────────────────────────
section "Step 3 — Dependencies"

info "Upgrading pip …"
pip install --upgrade pip -q

info "Installing requirements.txt …"
pip install -r requirements.txt -q
success "All packages installed."

# ── Step 4: .env setup ───────────────────────────────────────────
section "Step 4 — Environment / API Keys"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  echo ""
  echo -e "${YELLOW}╔══════════════════════════════════════════════════╗${RESET}"
  echo -e "${YELLOW}║  TMDB API Key (optional but recommended)         ║${RESET}"
  echo -e "${YELLOW}║  Get a free key at:                              ║${RESET}"
  echo -e "${YELLOW}║  https://www.themoviedb.org/settings/api         ║${RESET}"
  echo -e "${YELLOW}╚══════════════════════════════════════════════════╝${RESET}"
  echo ""
  read -p "  Paste your TMDB API key (or press Enter to skip): " TMDB_KEY

  if [ -n "$TMDB_KEY" ]; then
    # Works on macOS (BSD sed) and Linux (GNU sed)
    sed -i.bak "s/your_tmdb_api_key_here/$TMDB_KEY/" "$SCRIPT_DIR/.env"
    rm -f "$SCRIPT_DIR/.env.bak"
    success "TMDB API key saved to .env"
  else
    warn "Skipped — running without TMDB (no posters, no trending)."
    warn "You can add it later: edit .env and set TMDB_API_KEY=..."
  fi
else
  success ".env already exists."
fi

# ── Step 5: Train model ───────────────────────────────────────────
section "Step 5 — Training ML Model"

MOVIES_PKL="$SCRIPT_DIR/model/movies.pkl"
SIMILARITY_PKL="$SCRIPT_DIR/model/similarity.pkl"

if [ -f "$MOVIES_PKL" ] && [ -f "$SIMILARITY_PKL" ]; then
  info "Model already trained. Skipping. (Delete model/*.pkl to retrain)"
else
  info "Running train.py — this may take 30-90 seconds …"
  python train.py
  success "Model trained and saved."
fi

# ── Step 6: Launch ────────────────────────────────────────────────
section "Step 6 — Launching CineML"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║                                                  ║${RESET}"
echo -e "${GREEN}║   🎬  CineML is starting…                        ║${RESET}"
echo -e "${GREEN}║                                                  ║${RESET}"
echo -e "${GREEN}║   Open in browser:  http://localhost:5000        ║${RESET}"
echo -e "${GREEN}║   Stop server:      Ctrl + C                     ║${RESET}"
echo -e "${GREEN}║                                                  ║${RESET}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# Auto-open in browser after a short delay
(sleep 2 && open "http://localhost:5000") &

python app.py
