#!/usr/bin/env bash

echo "=== Setting up Phase 6: Multi-Agent System ==="

if command -v python3 &>/dev/null; then
    PY=python3
else
    PY=python
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PY -m venv venv
fi

# Activate venv (Windows + Git Bash compatible)
if [[ -f "venv/Scripts/activate" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "Installing dependencies..."
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=== Starting Streamlit App ==="
echo ""

python -m streamlit run app.py
