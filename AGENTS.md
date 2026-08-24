# AstroSim: Differentiable Multi-Agent Rocket Simulator

A Python project that builds a differentiable, vectorized, interactive rocket-landing
simulator in progressive phases. See `README.md` for the full project overview.

## Cursor Cloud specific instructions

### Services / entry points

This repo is a set of standalone Python scripts and Streamlit web apps, organized by phase.
There is no shared package; each phase is run directly from its own directory.

- `Phase_1_Differentiable_Physics/differentiable_physics.py` — CLI script (gradient descent demo; writes `phase1_loss_curve.png`).
- `Phase_2_Vectorized_Simulation/vectorized_simulation.py` — CLI script (`torch.vmap` fleet sim).
- `Phase_3_Streamlit_App`, `Phase_4_Optimized_Gui`, `Phase_5_Integrated_App`, `Phase_6_Multi_Agent` — Streamlit apps (`app.py`).
- `Phase_5_Integrated_App/app.py` is the primary integrated app and is auto-started by
  `.cursor/environment.json` in a terminal on port 8501.

### Running

- Streamlit apps: `python3 -m streamlit run <Phase_dir>/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`.
  Only one app can bind port 8501 at a time; use a different `--server.port` to run a second phase concurrently.
- Health check: `curl -s http://localhost:8501/_stcore/health` returns `ok`; the page root returns HTTP 200.
- CLI scripts: run from inside the phase directory (they save output PNGs to the current working directory).
- The per-phase `run.sh` scripts create a local `venv/` and reinstall deps; this is NOT needed in the cloud
  environment (dependencies are installed globally via `.cursor/install.sh`). Prefer invoking `python3 -m streamlit ...` directly.

### Dependencies / non-obvious notes

- Dependencies (`torch` CPU wheel, `numpy`, `matplotlib`, `streamlit`) are installed with `pip install --user`
  by `.cursor/install.sh`. There is no lockfile; `requirements.txt` is unpinned.
- Matplotlib must use a non-interactive backend in this headless environment. `.cursor/install.sh` writes
  `backend: Agg` to `~/.config/matplotlib/matplotlibrc`; do not switch to an interactive backend.
- There is no lint config and no automated test suite in this repo. A syntax smoke check is
  `python3 -m py_compile <file>.py`.
- The physics in these demos is an intentionally simple Euler integrator and is numerically unstable for the
  default parameters: losses can grow into the hundreds of thousands and final altitudes can go strongly
  negative. This divergence is expected demo behavior, not an environment bug.
