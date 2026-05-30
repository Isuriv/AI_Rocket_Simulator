#!/bin/bash

echo "=== Setting up Phase 4: Gradient Optimization GUI ==="

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py --server.port 8502
