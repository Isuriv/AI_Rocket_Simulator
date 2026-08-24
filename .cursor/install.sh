#!/usr/bin/env bash
set -euo pipefail

# Idempotent Cloud Agent / local bootstrap for AstroSim.
# Uses --user so it does not require python3-venv or a project virtualenv.

python3 -m pip install --user --upgrade pip
python3 -m pip install --user numpy matplotlib streamlit
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu

# Headless matplotlib: Phase 1 saves a PNG and Streamlit renders figures.
mkdir -p "${HOME}/.config/matplotlib"
printf 'backend: Agg\n' > "${HOME}/.config/matplotlib/matplotlibrc"
