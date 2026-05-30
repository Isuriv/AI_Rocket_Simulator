import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

st.set_page_config(page_title="Rocket Optimizer", layout="wide")
st.title("🚀 Phase 4: Gradient Optimization + Interactive GUI")
st.write("Run simulations and optimize landing using **differentiable physics** (gradients).")

# ====================== PHYSICS ======================
GRAVITY = 9.81
DT = 0.05
MAX_THRUST = 20.0
MASS = 10.0
INERTIA = 40.0
NUM_STEPS = 180

def simulate_rocket(initial_state, thrust, torque):
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

        trajectory.append([x.item(), y.item()])

    final_state = torch.stack([x, y, vx, vy, theta, omega, fuel])
    return final_state, torch.tensor(trajectory)

def compute_loss(final_state):
    x, y, vx, vy, theta, omega, fuel = final_state
    return (x**2 + y**2) * 1.5 + (vx**2 + vy**2) + (theta**2) * 2 + torch.relu(-y) * 50

# ====================== SIDEBAR ======================
st.sidebar.header("Initial Conditions")
initial_x = st.sidebar.slider("Initial X", -5.0, 5.0, 1.5)
initial_y = st.sidebar.slider("Initial Y", 15.0, 40.0, 28.0)
initial_angle = st.sidebar.slider("Initial Angle", -0.5, 0.5, 0.15)

initial_state = torch.tensor([initial_x, initial_y, 0.3, -0.8, initial_angle, 0.0, 45.0])

# ====================== MANUAL SIMULATION ======================
st.header("1. Manual Simulation")

col1, col2 = st.columns(2)
with col1:
    thrust = st.slider("Thrust", 5.0, 20.0, 12.0)
with col2:
    torque = st.slider("Torque", -2.0, 2.0, 0.2)

if st.button("Run Manual Simulation"):
    final_state, trajectory = simulate_rocket(initial_state, torch.tensor(thrust), torch.tensor(torque))
    
    fig, ax = plt.subplots()
    traj = trajectory.numpy()
    ax.plot(traj[:, 0], traj[:, 1])
    ax.axhline(y=0, color='green', linestyle='--')
    ax.scatter([0], [0], color='orange', s=150, zorder=5, label="Landing Pad")
    ax.set_title("Rocket Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Altitude")
    st.pyplot(fig)

    st.write(f"**Final Position:** x={final_state[0].item():.2f}, y={final_state[1].item():.2f}")
    st.write(f"**Landing Loss:** {compute_loss(final_state).item():.2f}")

st.markdown("---")

# ====================== GRADIENT OPTIMIZATION ======================
st.header("2. Gradient-Based Optimization")

st.write("Optimize thrust and torque automatically using **gradients** through the physics simulation.")

if st.button("Run Gradient Optimization"):
    with st.spinner("Optimizing..."):
        thrust_opt = torch.tensor(12.0, requires_grad=True)
        torque_opt = torch.tensor(0.2, requires_grad=True)

        optimizer = optim.Adam([thrust_opt, torque_opt], lr=0.5)

        losses = []
        for _ in range(80):
            optimizer.zero_grad()
            final_state, _ = simulate_rocket(initial_state, thrust_opt, torque_opt)
            loss = compute_loss(final_state)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Final results
        final_state, trajectory = simulate_rocket(initial_state, thrust_opt, torque_opt)

        st.success("Optimization Complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Optimized Thrust", f"{thrust_opt.item():.2f}")
            st.metric("Optimized Torque", f"{torque_opt.item():.2f}")
        with col2:
            st.metric("Final Loss", f"{compute_loss(final_state).item():.2f}")

        # Plot trajectory after optimization
        fig, ax = plt.subplots()
        traj = trajectory.numpy()
        ax.plot(traj[:, 0], traj[:, 1], color='blue')
        ax.axhline(y=0, color='green', linestyle='--')
        ax.scatter([0], [0], color='orange', s=150, zorder=5)
        ax.set_title("Optimized Rocket Trajectory")
        st.pyplot(fig)

        # Loss curve
        fig2, ax2 = plt.subplots()
        ax2.plot(losses)
        ax2.set_title("Optimization Loss Curve")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        st.pyplot(fig2)
