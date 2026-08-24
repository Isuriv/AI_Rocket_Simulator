import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

st.set_page_config(page_title="AstroSim - Integrated Rocket Simulator", layout="wide")
st.title("🚀 AstroSim - Phase 5: Integrated Rocket Simulator")
st.caption("Differentiable Physics + Vectorization + Interactive Web GUI")

# ====================== SHARED PHYSICS ======================
GRAVITY = 9.81
DT = 0.05
MAX_THRUST = 20.0
MASS = 10.0
INERTIA = 40.0
NUM_STEPS = 180

def simulate_single_rocket(initial_state, thrust, torque):
    x, y, vx, vy, theta, omega, fuel = initial_state
    trajectory = []

    for _ in range(NUM_STEPS):
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

        # Keep positions as tensors so torch.vmap can batch this function.
        trajectory.append(torch.stack([x, y]))

    final_state = torch.stack([x, y, vx, vy, theta, omega, fuel])
    return final_state, torch.stack(trajectory)

def simulate_fleet(initial_states, thrusts, torques):
    """Vectorized simulation for multiple rockets"""
    batched_sim = torch.vmap(simulate_single_rocket, in_dims=(0, 0, 0))
    final_states, trajectories = batched_sim(initial_states, thrusts, torques)
    return final_states, trajectories

def compute_loss(final_state):
    x, y, vx, vy, theta, omega, fuel = final_state.unbind(-1)
    return (x**2 + y**2) * 1.5 + (vx**2 + vy**2) + theta**2 * 2 + torch.relu(-y) * 50

# ====================== SIDEBAR ======================
st.sidebar.header("Global Settings")
num_rockets = st.sidebar.slider("Number of Rockets (Fleet)", 1, 50, 5)

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["Single Rocket", "Fleet Simulation", "Gradient Optimization"])

# ====================== TAB 1: SINGLE ROCKET ======================
with tab1:
    st.header("Single Rocket Simulation")

    col1, col2 = st.columns(2)
    with col1:
        thrust = st.slider("Thrust", 5.0, 20.0, 12.0, key="single_thrust")
        torque = st.slider("Torque", -2.0, 2.0, 0.2, key="single_torque")
    with col2:
        init_x = st.slider("Initial X", -5.0, 5.0, 1.5, key="single_x")
        init_y = st.slider("Initial Y", 15.0, 40.0, 28.0, key="single_y")

    initial_state = torch.tensor([init_x, init_y, 0.3, -0.8, 0.15, 0.0, 45.0])

    if st.button("Run Single Rocket Simulation", key="run_single"):
        final_state, trajectory = simulate_single_rocket(initial_state, torch.tensor(thrust), torch.tensor(torque))

        fig, ax = plt.subplots()
        traj = trajectory.numpy()
        ax.plot(traj[:, 0], traj[:, 1], label="Trajectory")
        ax.axhline(y=0, color='green', linestyle='--')
        ax.scatter([0], [0], color='orange', s=120, zorder=5, label="Landing Pad")
        ax.set_title("Single Rocket Trajectory")
        ax.legend()
        st.pyplot(fig)

        st.write(f"**Final Position:** x={final_state[0].item():.2f}, y={final_state[1].item():.2f}")
        st.write(f"**Landing Loss:** {compute_loss(final_state).item():.2f}")

# ====================== TAB 2: FLEET SIMULATION ======================
with tab2:
    st.header("Fleet Simulation (Vectorized)")

    if st.button("Simulate Fleet"):
        # Create random initial states for the fleet
        initial_states = torch.zeros((num_rockets, 7))
        initial_states[:, 0] = torch.randn(num_rockets) * 4
        initial_states[:, 1] = 25 + torch.randn(num_rockets) * 6
        initial_states[:, 4] = torch.randn(num_rockets) * 0.3
        initial_states[:, 6] = 45.0

        thrusts = torch.rand(num_rockets) * 12 + 6
        torques = torch.randn(num_rockets) * 0.4

        final_states, trajectories = simulate_fleet(initial_states, thrusts, torques)

        # Plot all trajectories
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(num_rockets):
            traj = trajectories[i].numpy()
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.6)
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2)
        ax.scatter([0], [0], color='orange', s=150, zorder=5, label="Landing Pad")
        ax.set_title(f"Fleet Simulation ({num_rockets} Rockets)")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Altitude")
        st.pyplot(fig)

        avg_loss = compute_loss(final_states).mean().item()
        st.write(f"**Average Landing Loss across fleet:** {avg_loss:.2f}")

# ====================== TAB 3: GRADIENT OPTIMIZATION ======================
with tab3:
    st.header("Gradient-Based Optimization")

    opt_init_x = st.slider("Initial X (Optimization)", -5.0, 5.0, 1.5, key="opt_x")
    opt_init_y = st.slider("Initial Y (Optimization)", 15.0, 40.0, 28.0, key="opt_y")

    opt_initial_state = torch.tensor([opt_init_x, opt_init_y, 0.3, -0.8, 0.15, 0.0, 45.0])

    if st.button("Run Gradient Optimization"):
        with st.spinner("Optimizing parameters..."):
            thrust_opt = torch.tensor(12.0, requires_grad=True)
            torque_opt = torch.tensor(0.2, requires_grad=True)

            optimizer = optim.Adam([thrust_opt, torque_opt], lr=0.4)
            losses = []

            for _ in range(100):
                optimizer.zero_grad()
                final_state, _ = simulate_single_rocket(opt_initial_state, thrust_opt, torque_opt)
                loss = compute_loss(final_state)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # Final optimized simulation
            final_opt, trajectory_opt = simulate_single_rocket(opt_initial_state, thrust_opt, torque_opt)

            st.success("Optimization Complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimized Thrust", f"{thrust_opt.item():.2f}")
                st.metric("Optimized Torque", f"{torque_opt.item():.2f}")
            with col2:
                st.metric("Final Loss", f"{compute_loss(final_opt).item():.2f}")

            # Plot optimized trajectory
            fig, ax = plt.subplots()
            traj = trajectory_opt.numpy()
            ax.plot(traj[:, 0], traj[:, 1], color='blue')
            ax.axhline(y=0, color='green', linestyle='--')
            ax.scatter([0], [0], color='orange', s=120, zorder=5)
            ax.set_title("Optimized Trajectory")
            st.pyplot(fig)

            # Loss curve
            fig2, ax2 = plt.subplots()
            ax2.plot(losses)
            ax2.set_title("Optimization Progress (Loss Curve)")
            st.pyplot(fig2)
