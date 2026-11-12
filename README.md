# AstroSim: Differentiable Multi-Agent Rocket Simulator

A 2D rocket landing simulator written in PyTorch. The physics is differentiable, so thrust and torque can be tuned by backpropagating a landing loss through all 180 simulation steps, and it is vectorized with `torch.vmap` so many rockets run in one call.

## Benchmark: `torch.vmap` vs a plain Python loop

Same physics function (`simulate_single_rocket`, 180 steps, forward pass) called once per rocket in a Python loop versus once for the whole batch through `torch.vmap`.

| Rockets | Plain loop | `torch.vmap` | Speedup | Peak memory (loop) | Peak memory (vmap) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.0 ms | 12.7 ms | 0.5x | <0.1 MB | <0.1 MB |
| 10 | 63.3 ms | 17.7 ms | 3.6x | <0.1 MB | <0.1 MB |
| 100 | 655.9 ms | 13.9 ms | 47.3x | <0.1 MB | <0.1 MB |
| 1,000 | 6.54 s | 14.4 ms | 454.7x | <0.1 MB | <0.1 MB |
| 10,000 | 66.00 s | 24.0 ms | 2755.0x | 5.2 MB | 1.8 MB |

Hardware: Intel Xeon (cloud VM, CPU only, 4 PyTorch threads), PyTorch 2.15.0+cpu, Python 3.12.3, Linux. Times are medians of repeated runs (5 runs, except 3 for the 1,000-rocket loop and 1 for the 10,000-rocket loop). Memory is peak process RSS growth during the timed runs, on top of a ~262 MB baseline for Python, PyTorch and the inputs. Loop and vmap final states match exactly (max difference 0.0).

For a single rocket, vmap is slower because of its batching overhead. From 10 rockets up it wins, and its time barely grows between 1 and 10,000 rockets on this CPU.

Reproduce it (takes about 2 minutes, most of it in the 10,000-rocket loop):

```bash
cd Phase_2_Vectorized_Simulation
python benchmark.py --markdown benchmark_results.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Isuriv/AI_Rocket_Simulator.git
cd AI_Rocket_Simulator
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run a phase

Command-line phases:

```bash
cd Phase_1_Differentiable_Physics
python differentiable_physics.py   # optimizes thrust/torque, saves phase1_loss_curve.png
```

```bash
cd Phase_2_Vectorized_Simulation
python vectorized_simulation.py    # simulates 128 rockets with torch.vmap
```

Streamlit phases (each opens at http://localhost:8501):

```bash
cd Phase_4_Optimized_Gui
streamlit run app.py
```

Each phase folder also has a `run.sh` that creates its own virtual environment and runs that phase.

## Project phases

| Folder | What it does |
| --- | --- |
| `Phase_1_Differentiable_Physics/` | Single-rocket simulator. Optimizes constant thrust and torque with Adam by backpropagating a landing loss through the simulation. |
| `Phase_2_Vectorized_Simulation/` | The same physics batched over many rockets with `torch.vmap`, plus `benchmark.py` for the table above. |
| `Phase_3_Streamlit_App/` | Streamlit page with sliders for thrust, torque and initial conditions, and a trajectory plot. |
| `Phase_4_Optimized_Gui/` | Adds a one-click gradient optimization with the loss curve and optimized trajectory. |
| `Phase_5_Integrated_App/` | Tabs for single rocket, fleet and gradient optimization in one app. |
| `Phase_6_Multi_Agent/` | Several rockets steering to their own targets with distance-based collision avoidance (heuristic controller, not differentiable). |

## Project structure

```
AI_Rocket_Simulator/
├── Phase_1_Differentiable_Physics/
├── Phase_2_Vectorized_Simulation/
├── Phase_3_Streamlit_App/
├── Phase_4_Optimized_Gui/
├── Phase_5_Integrated_App/
├── Phase_6_Multi_Agent/
├── requirements.txt
└── README.md
```

## Tech stack

PyTorch (simulation, autograd, `torch.vmap`), Streamlit (web UI), NumPy and Matplotlib (data and plots).
