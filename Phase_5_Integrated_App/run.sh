#!/usr/bin/env bash

echo "=== Setting up Phase 5: Integrated Web Application ==="

# Detect python command
if command -v python3 &>/dev/null; then
    PY=python3
else
    PY=python
fi

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PY -m venv venv
fi

# Activate venv (Windows Git Bash compatible)
if [[ -f "venv/Scripts/activate" ]]; then
    source venv/Scripts/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
else
    echo "ERROR: Could not find venv activation script"
    exit 1
fi

echo "Installing/updating dependencies..."
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=== Launching Streamlit App ==="
echo ""

python -m streamlit run app.py
