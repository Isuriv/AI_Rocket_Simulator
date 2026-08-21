#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --user --upgrade pip
python3 -m pip install --user numpy matplotlib streamlit
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu

mkdir -p "$HOME/.config/matplotlib"
printf 'backend: Agg\n' > "$HOME/.config/matplotlib/matplotlibrc"
