#!/usr/bin/env python3
"""
Phase 1: Core Differentiable Rocket Physics (PyTorch)

This file demonstrates how to build a rocket landing simulation
where the entire physics is differentiable using PyTorch.

Key Goal: Allow gradients to flow through the simulation so we can
optimize parameters (like thrust) directly using gradient descent.
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================== PHYSICS CONSTANTS ======================
GRAVITY = 9.81          # Acceleration due to gravity (m/s^2)
DT = 0.05               # Time step for simulation (seconds)
MAX_THRUST = 20.0       # Maximum thrust the rocket can produce
MASS = 10.0             # Mass of the rocket (kg)
INERTIA = 40.0          # Moment of inertia (how hard it is to rotate)
NUM_STEPS = 180         # Number of simulation steps


def simulate_rocket(initial_state, thrust, torque):
    """
    Runs the rocket simulation for a fixed number of steps.
    
    This function is fully differentiable because all operations
    use PyTorch tensors and differentiable math operations.
    
    Args:
        initial_state: Tensor of shape [7] -> [x, y, vx, vy, theta, omega, fuel]
        thrust: Scalar tensor (can have requires_grad=True)
        torque: Scalar tensor (can have requires_grad=True)
    
    Returns:
        final_state: The state after simulation
    """
    # Unpack initial state
    x, y, vx, vy, theta, omega, fuel = initial_state

    # Run simulation for NUM_STEPS time steps
    for t in range(NUM_STEPS):
        # Calculate thrust components based on current angle (thrust vectoring)
        thrust_x = thrust * torch.sin(theta)
        thrust_y = thrust * torch.cos(theta)

        # Calculate acceleration
        ax = thrust_x / MASS                    # Horizontal acceleration
        ay = thrust_y / MASS - GRAVITY          # Vertical acceleration (gravity pulls down)

        # Update velocities
        vx = vx + ax * DT
        vy = vy + ay * DT

        # Update positions
        x = x + vx * DT
        y = y + vy * DT

        # Rotational dynamics
        alpha = torque / INERTIA                # Angular acceleration
        omega = omega + alpha * DT              # Update angular velocity
        theta = theta + omega * DT              # Update angle

        # Simple fuel consumption model
        fuel = fuel - (thrust / MAX_THRUST) * 0.25 * DT

    # Stack all final values into one tensor
    final_state = torch.stack([x, y, vx, vy, theta, omega, fuel])
    return final_state


def compute_landing_loss(final_state):
    """
    Calculates how "bad" the landing was.
    Lower loss = better landing.
    """
    x, y, vx, vy, theta, omega, fuel = final_state

    position_error = x**2 + y**2           # Want to be close to (0, 0)
    velocity_error = vx**2 + vy**2         # Want low speed when landing
    angle_error = theta**2                 # Want to be upright
    ground_penalty = torch.relu(-y) * 50   # Big penalty if we go below ground

    # Combine all errors into one loss value
    loss = position_error * 1.5 + velocity_error + angle_error * 2 + ground_penalty
    return loss


def main():
    print("=== Phase 1: Differentiable Rocket Physics ===\n")

    # Define starting conditions of the rocket
    # [x position, y position, x velocity, y velocity, angle, angular velocity, fuel]
    initial_state = torch.tensor(
        [1.5, 28.0, 0.3, -0.8, 0.15, 0.0, 45.0],
        dtype=torch.float32,
        device=device
    )

    # Make thrust and torque learnable (we will optimize them using gradients)
    thrust = torch.tensor(11.0, device=device, requires_grad=True)
    torque = torch.tensor(0.3, device=device, requires_grad=True)

    # Use Adam optimizer to adjust thrust and torque
    optimizer = optim.Adam([thrust, torque], lr=0.4)

    print("Optimizing thrust and torque using gradients...\n")

    losses = []  # Store loss values to plot later

    # Run optimization for 100 steps
    for step in range(100):
        optimizer.zero_grad()                    # Clear previous gradients

        # Run the simulation (this is where differentiability matters)
        final_state = simulate_rocket(initial_state, thrust, torque)

        # Calculate how bad the landing was
        loss = compute_landing_loss(final_state)

        # Compute gradients (this is the key part of differentiability)
        loss.backward()

        # Update thrust and torque based on gradients
        optimizer.step()

        losses.append(loss.item())

        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():8.1f} | Thrust: {thrust.item():.2f}")

    print(f"\n✅ Final Optimized Thrust: {thrust.item():.3f}")
    print(f"✅ Final Optimized Torque: {torque.item():.3f}")

    # Show final landing result
    with torch.no_grad():
        final_state = simulate_rocket(initial_state, thrust, torque)
        print(f"\nFinal Position → x: {final_state[0].item():.2f}, y: {final_state[1].item():.2f}")

    # Plot how the loss decreased during optimization
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Loss During Gradient-Based Optimization")
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("phase1_loss_curve.png", dpi=150)
    print("\n📊 Loss curve saved as phase1_loss_curve.png")


if __name__ == "__main__":
    main()
