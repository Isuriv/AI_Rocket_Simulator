#!/bin/bash

echo "=== Setting up Phase 3: Streamlit Web Interface ==="

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting Streamlit app..."
streamlit run app.py --server.port 8501
