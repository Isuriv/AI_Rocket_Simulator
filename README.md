# AstroSim: Differentiable Multi-Agent Rocket Simulator

A progressive project that builds a **fully differentiable, vectorized, and interactive rocket landing simulator** — from core physics to a web-based GUI.

This project demonstrates strong skills in **physics simulation, automatic differentiation, vectorized computing, and modern web interfaces** — highly relevant for roles in aerospace AI and simulation (e.g. SpaceXAI).

## Project Overview

The goal is to simulate rocket landings using **differentiable physics**, allowing both traditional control and gradient-based optimization. The project is built in progressive phases, culminating in a full interactive web application.

## Project Phases

### Phase 1: Core Differentiable Physics
- Built a 2D rocket landing simulator in **PyTorch**
- Made the entire physics simulation **differentiable**
- Demonstrated gradient-based optimization of thrust and torque

### Phase 2: Vectorized Multi-Rocket Simulation
- Used `torch.vmap` to simulate **hundreds of rockets in parallel**
- Maintained full differentiability across the batch
- Showed how to scale simulation efficiently

### Phase 3: Streamlit Web Interface
- Created an interactive web GUI using **Streamlit**
- Added sliders for thrust, torque, and initial conditions
- Real-time trajectory visualization

### Phase 4: Gradient Optimization in GUI
- Integrated gradient-based optimization directly into the web interface
- Users can run optimization with one click
- Visualized loss curves and optimized trajectories

### Phase 5: Full Integrated Web Application (Final Phase)

**Goal**: Blend all previous phases into one cohesive, production-ready web application.

**Planned Features:**
- Clean, tabbed interface (Simulation, Optimization, Fleet View)
- Support for simulating **multiple rockets** simultaneously
- Toggle between **manual control** and **gradient-based optimization**
- Real-time animated trajectory visualization
- Comparison mode: Manual vs Optimized landing
- Performance metrics and loss tracking
- Scalable backend using vectorized differentiable physics
- Option to export results and simulation data

This final phase combines:
- Differentiable physics (Phase 1)
- Vectorized computation (Phase 2)
- Modern web interface (Phase 3)
- Interactive optimization (Phase 4)

---

## Tech Stack

- **PyTorch** — Differentiable physics and automatic differentiation
- **Streamlit** — Interactive web interface
- **NumPy + Matplotlib** — Data handling and visualization
- **Python** — Core language

## Why This Project Matters

This project showcases:
- Deep understanding of **physics-informed machine learning**
- Ability to build **scalable, differentiable simulations**
- Experience with **modern Python tooling** and web interfaces
- End-to-end thinking: from low-level physics to user-facing applications

These skills are directly applicable to aerospace simulation, autonomous systems, and AI-driven engineering at companies like SpaceX.

## Project Structure
rocket-simulator/
├── phase1_differentiable_physics/
├── phase2_vectorized_simulation/
├── phase3_streamlit_app/
├── phase4_optimized_gui/
├── phase5_integrated_app/          # ← Final integrated version
├── requirements.txt
└── README.md

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rocket-simulator.git
cd rocket-simulator
```
### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 3. Run individual phases
```bash
cd phase4_optimized_gui
chmod +x run.sh
./run.sh
```
