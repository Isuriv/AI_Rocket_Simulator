import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rocket Landing Simulator", layout="wide")
st.title("🚀 Phase 3: Interactive Rocket Landing Simulator")
st.write("This is the beginning of the web GUI. Built with Streamlit + PyTorch (Differentiable Physics)")

# ====================== PHYSICS (from Phase 1) ======================
GRAVITY = 9.81
DT = 0.05
MAX_THRUST = 20.0
MASS = 10.0
INERTIA = 40.0
NUM_STEPS = 180

def simulate_rocket(initial_state, thrust, torque):
    x, y, vx, vy, theta, omega, fuel = initial_state
    trajectory = []

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
        trajectory.append([x.item(), y.item()])

    final_state = torch.stack([x, y, vx, vy, theta, omega, fuel])
    return final_state, torch.tensor(trajectory)

def compute_loss(final_state):
    x, y, vx, vy, theta, omega, fuel = final_state
    return (x**2 + y**2) * 1.5 + (vx**2 + vy**2) + theta**2 * 2 + torch.relu(-y) * 50

# ====================== SIDEBAR CONTROLS ======================
st.sidebar.header("Simulation Parameters")

thrust = st.sidebar.slider("Thrust", 5.0, 20.0, 12.0, 0.5)
torque = st.sidebar.slider("Torque", -2.0, 2.0, 0.2, 0.1)

initial_x = st.sidebar.slider("Initial X Position", -5.0, 5.0, 1.5, 0.5)
initial_y = st.sidebar.slider("Initial Y Position", 15.0, 40.0, 28.0, 1.0)
initial_angle = st.sidebar.slider("Initial Angle (radians)", -0.5, 0.5, 0.15, 0.05)

if st.sidebar.button("Run Simulation"):
    initial_state = torch.tensor([initial_x, initial_y, 0.3, -0.8, initial_angle, 0.0, 45.0])

    final_state, trajectory = simulate_rocket(initial_state, torch.tensor(thrust), torch.tensor(torque))

    # Plot trajectory
    fig, ax = plt.subplots()
    traj_np = trajectory.numpy()
    ax.plot(traj_np[:, 0], traj_np[:, 1], label="Rocket Path")
    ax.axhline(y=0, color='green', linestyle='--', label="Ground")
    ax.scatter([0], [0], color='orange', s=100, label="Landing Pad", zorder=5)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position (Altitude)")
    ax.set_title("Rocket Trajectory")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # Show results
    st.subheader("Final Results")
    st.write(f"**Final X:** {final_state[0].item():.2f}")
    st.write(f"**Final Y:** {final_state[1].item():.2f}")
    st.write(f"**Final Angle:** {np.rad2deg(final_state[4].item()):.1f}°")
    st.write(f"**Landing Loss:** {compute_loss(final_state).item():.2f}")

st.markdown("---")
st.info("This is Phase 3. In future phases we will add gradient optimization, multiple rockets, and more advanced controls.")