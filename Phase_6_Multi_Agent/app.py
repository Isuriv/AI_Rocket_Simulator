import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Phase 6: Multi-Agent Rocket Simulator", layout="wide")
st.title("🚀 Phase 6: Multi-Agent Rocket Features")
st.write("Multiple rockets with different goals + basic collision avoidance")

# ====================== PHYSICS ======================
GRAVITY = 9.81
DT = 0.05
MAX_THRUST = 18.0
MASS = 10.0
NUM_STEPS = 200

def simulate_multi_agent(num_agents=5, avoidance_strength=0.8):
    # Initialize agents with different starting positions and slight goal variations
    positions = torch.zeros((num_agents, 2))
    velocities = torch.zeros((num_agents, 2))
    angles = torch.zeros(num_agents)
    angular_vel = torch.zeros(num_agents)
    thrusts = torch.ones(num_agents) * 12.0
    torques = torch.zeros(num_agents)

    # Different starting X positions and slight goal offsets
    positions[:, 0] = torch.linspace(-6, 6, num_agents)
    positions[:, 1] = 30 + torch.randn(num_agents) * 3

    # Each rocket has a slightly different target X position
    targets = torch.linspace(-3, 3, num_agents)

    trajectory_history = [positions.clone()]

    for step in range(NUM_STEPS):
        # Simple thrust toward their individual targets + gravity compensation
        for i in range(num_agents):
            dx = targets[i] - positions[i, 0]
            desired_angle = torch.clamp(dx * 0.1, -0.6, 0.6)
            angles[i] = 0.7 * angles[i] + 0.3 * desired_angle

            # Basic collision avoidance
            for j in range(num_agents):
                if i != j:
                    diff = positions[i] - positions[j]
                    dist = torch.norm(diff) + 1e-6
                    if dist < 3.0:
                        avoidance = (diff / dist) * avoidance_strength
                        velocities[i] += avoidance * 0.3

            # Apply thrust in direction of current angle
            thrust_x = thrusts[i] * torch.sin(angles[i])
            thrust_y = thrusts[i] * torch.cos(angles[i])

            ax = thrust_x / MASS
            ay = thrust_y / MASS - GRAVITY

            velocities[i, 0] += ax * DT
            velocities[i, 1] += ay * DT

            positions[i] += velocities[i] * DT

        trajectory_history.append(positions.clone())

    return torch.stack(trajectory_history), targets

# ====================== UI ======================
st.sidebar.header("Multi-Agent Settings")
num_agents = st.sidebar.slider("Number of Rockets", 2, 12, 6)
avoidance = st.sidebar.slider("Collision Avoidance Strength", 0.0, 2.0, 0.8, 0.1)

if st.button("Run Multi-Agent Simulation"):
    with st.spinner("Simulating multiple rockets with collision avoidance..."):
        trajectories, targets = simulate_multi_agent(num_agents, avoidance)

        fig, ax = plt.subplots(figsize=(10, 7))

        for i in range(num_agents):
            traj = trajectories[:, i, :].numpy()
            ax.plot(traj[:, 0], traj[:, 1], label=f"Rocket {i+1}", alpha=0.75)

        # Draw targets
        for i, target_x in enumerate(targets):
            ax.scatter([target_x], [0], marker='x', s=100, linewidths=2)

        ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label="Ground")
        ax.set_title(f"Multi-Agent Simulation ({num_agents} Rockets)")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Altitude")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        st.success("Simulation complete! Each rocket tries to reach its own target while avoiding others.")
        st.info("Note: This is a simplified multi-agent demo with basic goal-seeking + repulsion-based collision avoidance.")
