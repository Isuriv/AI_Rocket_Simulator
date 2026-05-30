#!/usr/bin/env python3
"""
Phase 2: Vectorized Multi-Rocket Simulation using torch.vmap

This builds on Phase 1.
Instead of simulating one rocket at a time, we use torch.vmap
to simulate hundreds of rockets **in parallel** on GPU/CPU.

This is a key technique for scaling simulations efficiently.
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================== PHYSICS CONSTANTS ======================
GRAVITY = 9.81
DT = 0.05
MAX_THRUST = 20.0
MASS = 10.0
INERTIA = 40.0
NUM_STEPS = 180


def simulate_single_rocket(initial_state, thrust, torque):
    """
    Same physics as Phase 1, but designed to work with vmap.
    Simulates ONE rocket.
    """
    x, y, vx, vy, theta, omega, fuel = initial_state

    for t in range(NUM_STEPS):
        thrust_x = thrust * torch.sin(theta)
        thrust_y = thrust * torch.cos(theta)

        ax = thrust_x / MASS
        ay = thrust_y / MASS - GRAVITY

        vx = vx + ax * DT
        vy = vy + ay * DT
        x = x + vx * DT
        y = y + vy * DT

        alpha = torque / INERTIA
        omega = omega + alpha * DT
        theta = theta + omega * DT

        fuel = fuel - (thrust / MAX_THRUST) * 0.25 * DT

    return torch.stack([x, y, vx, vy, theta, omega, fuel])


def simulate_many_rockets(initial_states, thrusts, torques):
    """
    Vectorized version using torch.vmap.
    
    This runs the simulation for MANY rockets at the same time.
    
    Args:
        initial_states: Tensor of shape [N, 7]  (N rockets)
        thrusts: Tensor of shape [N]
        torques: Tensor of shape [N]
    
    Returns:
        final_states: Tensor of shape [N, 7]
    """
    # vmap applies simulate_single_rocket across the first dimension
    batched_simulate = torch.vmap(simulate_single_rocket, in_dims=(0, 0, 0))
    
    final_states = batched_simulate(initial_states, thrusts, torques)
    return final_states


def main():
    print("=== Phase 2: Vectorized Multi-Rocket Simulation ===\n")

    num_rockets = 128  # Simulate 128 rockets at once

    # Create random initial states for many rockets
    initial_states = torch.zeros((num_rockets, 7), device=device)
    initial_states[:, 0] = torch.randn(num_rockets, device=device) * 3      # x position
    initial_states[:, 1] = 25 + torch.randn(num_rockets, device=device) * 5 # y position
    initial_states[:, 2] = torch.randn(num_rockets, device=device) * 0.5    # vx
    initial_states[:, 3] = -1.0 + torch.randn(num_rockets, device=device) * 0.3
    initial_states[:, 4] = torch.randn(num_rockets, device=device) * 0.2    # angle
    initial_states[:, 6] = 40.0                                             # fuel

    # Random starting thrust and torque for each rocket
    thrusts = torch.rand(num_rockets, device=device) * 15 + 5
    torques = torch.randn(num_rockets, device=device) * 0.5

    print(f"Simulating {num_rockets} rockets in parallel using torch.vmap...\n")

    # Run vectorized simulation
    final_states = simulate_many_rockets(initial_states, thrusts, torques)

    # Calculate average landing quality
    avg_x = final_states[:, 0].mean().item()
    avg_y = final_states[:, 1].mean().item()
    avg_loss = (final_states[:, 0]**2 + final_states[:, 1]**2).mean().item()

    print(f"Average final X position: {avg_x:.2f}")
    print(f"Average final Y position: {avg_y:.2f}")
    print(f"Average landing error (lower is better): {avg_loss:.2f}")

    print("\n✅ Phase 2 simulation completed successfully!")
    print("This shows we can efficiently simulate many rockets at once.")


if __name__ == "__main__":
    main()
