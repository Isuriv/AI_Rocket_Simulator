# Phase 1: Differentiable Single Rocket Landing Simulator (PyTorch)

**Goal of this phase**: Create a rocket landing physics simulator where the **entire simulation is differentiable**.

This means we can compute gradients through the physics (e.g. "How does changing the thrust at step 50 affect the final landing position?").

This is the foundation for building vectorized + multi-agent versions later.

## What This Demonstrates
- Physics simulation in PyTorch
- Automatic differentiation through time (simulation rollout)
- Gradient-based optimization on physical parameters
- Clean, educational code

This is the first step toward a much more impressive **vectorized + differentiable multi-agent rocket simulator**.

## Key Concepts
- `torch.autograd` tracks operations so we can backpropagate gradients.
- We treat physical quantities (position, velocity, angle) as tensors.
- We can optimize parameters (thrust, initial angle, etc.) directly using gradients.

## Files
- `differentiable_rocket.py` — Main code
- `README.md` — This file

## How to Run

```bash
pip install torch numpy matplotlib
python differentiable_rocket.py
```

## Next Phases (Roadmap)
- Phase 2: Vectorize to simulate many rockets in parallel (`torch.vmap`)
- Phase 3: Multi-agent behavior
- Phase 4: Combine with RL + gradient optimization
- Phase 5: Move to JAX for cleaner scientific computing style

---

Built collaboratively with Grok as part of a SpaceXAI portfolio project.