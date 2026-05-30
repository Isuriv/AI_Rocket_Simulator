#!/bin/bash

echo "=== Setting up Phase 1: Differentiable Rocket Physics ==="

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
echo "Running Phase 1..."
python differentiable_physics.py

echo "Done!"