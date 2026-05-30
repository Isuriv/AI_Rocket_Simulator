#!/bin/bash

echo "=== Setting up Phase 2: Vectorized Multi-Rocket Simulation ==="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch numpy matplotlib

# Run the simulation
echo "Running Phase 2..."
python vectorized_simulation.py

echo "Done!"