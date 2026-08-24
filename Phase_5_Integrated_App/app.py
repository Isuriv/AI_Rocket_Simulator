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
MASS = 10.0
INERTIA = 40.0
NUM_STEPS = 180

# Thrust-to-weight must exceed 1 for the rocket to be able to slow its descent and
# land. The rocket's weight is MASS * GRAVITY (~98 N), so the engine has to be able to
# out-push gravity. MAX_THRUST gives a thrust-to-weight ratio of ~2.
MAX_THRUST = 200.0
MAX_TORQUE = 20.0
HOVER_THRUST = MASS * GRAVITY  # ~98.1 N: thrust needed to exactly cancel gravity when upright


def _step(state, thrust, torque):
    """Advance the rocket one time step. Pure tensor ops so it is both differentiable
    and safe to run under torch.vmap (no Python-scalar conversions)."""
    x, y, vx, vy, theta, omega, fuel = state

    ax = thrust * torch.sin(theta) / MASS
    ay = thrust * torch.cos(theta) / MASS - GRAVITY

    vx = vx + ax * DT
    vy = vy + ay * DT
    x = x + vx * DT
    y = y + vy * DT

    omega = omega + (torque / INERTIA) * DT
    theta = theta + omega * DT
    fuel = fuel - (thrust / MAX_THRUST) * 0.25 * DT

    return torch.stack([x, y, vx, vy, theta, omega, fuel])


def simulate_single_rocket(initial_state, thrust, torque):
    """Simulate one rocket under a constant thrust/torque. Returns the final state and
    the (x, y) trajectory. vmap-safe: the trajectory is built with torch.stack instead
    of .item(), so this same function powers the vectorized fleet simulation."""
    state = initial_state
    trajectory = []
    for _ in range(NUM_STEPS):
        state = _step(state, thrust, torque)
        trajectory.append(state[:2])
    return state, torch.stack(trajectory)


def simulate_schedule(initial_state, thrust_seq, torque_seq):
    """Simulate one rocket under a time-varying thrust/torque schedule. This extra
    control authority is what lets gradient optimization actually land the rocket."""
    state = initial_state
    trajectory = []
    for t in range(NUM_STEPS):
        state = _step(state, thrust_seq[t], torque_seq[t])
        trajectory.append(state[:2])
    return state, torch.stack(trajectory)


def simulate_fleet(initial_states, thrusts, torques):
    """Vectorized simulation for multiple rockets."""
    batched_sim = torch.vmap(simulate_single_rocket, in_dims=(0, 0, 0))
    final_states, trajectories = batched_sim(initial_states, thrusts, torques)
    return final_states, trajectories


def compute_loss(final_state):
    """Landing score for a single final state (lower is better)."""
    x, y, vx, vy, theta, omega, fuel = final_state
    return (x**2 + y**2) * 1.5 + (vx**2 + vy**2) + theta**2 * 2 + torch.relu(-y) * 50


def landing_loss(final_state, trajectory, thrust_seq):
    """Multi-objective landing loss for schedule optimization: land at the pad (0, 0),
    with near-zero speed, upright, without dipping below ground, using modest thrust."""
    x, y, vx, vy, theta, omega, fuel = final_state
    position = x**2 + y**2
    velocity = vx**2 + vy**2
    attitude = theta**2 + 0.1 * omega**2
    underground = torch.relu(-trajectory[:, 1]).pow(2).sum()
    effort = 1e-5 * (thrust_seq**2).mean()
    return 5 * position + 3 * velocity + 18 * attitude + 5 * underground + effort


def clip_to_ground(traj_np):
    """Truncate a trajectory at the first point it reaches/passes the ground (y <= 0)
    so plots stop at the landing pad instead of continuing underground."""
    ys = traj_np[:, 1]
    below = np.where(ys <= 0)[0]
    if len(below) > 0:
        return traj_np[: below[0] + 1]
    return traj_np


def landing_verdict(distance, speed, tilt):
    if distance < 1.0 and speed < 1.0 and abs(tilt) < 0.35:
        return "success", "Soft landing on the pad! 🎯"
    if distance < 3.0 and speed < 3.0:
        return "warning", "Rough landing — close, but not a clean touchdown."
    return "error", "Crash — the rocket missed the pad or came in too hot."


# ====================== SIDEBAR ======================
st.sidebar.header("Global Settings")
num_rockets = st.sidebar.slider("Number of Rockets (Fleet)", 1, 50, 5)
st.sidebar.caption(f"Hover thrust ≈ {HOVER_THRUST:.0f} N (thrust needed to cancel gravity).")

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["Single Rocket", "Fleet Simulation", "Gradient Optimization"])

# ====================== TAB 1: SINGLE ROCKET ======================
with tab1:
    st.header("Single Rocket Simulation")
    st.caption("Manual constant-control sandbox. Thrust above the hover value slows the descent.")

    col1, col2 = st.columns(2)
    with col1:
        thrust = st.slider("Thrust", 0.0, MAX_THRUST, HOVER_THRUST, key="single_thrust")
        torque = st.slider("Torque", -MAX_TORQUE, MAX_TORQUE, 0.0, key="single_torque")
    with col2:
        init_x = st.slider("Initial X", -5.0, 5.0, 1.5, key="single_x")
        init_y = st.slider("Initial Y", 15.0, 40.0, 28.0, key="single_y")

    initial_state = torch.tensor([init_x, init_y, 0.3, -0.8, 0.15, 0.0, 45.0])

    if st.button("Run Single Rocket Simulation", key="run_single"):
        final_state, trajectory = simulate_single_rocket(
            initial_state, torch.tensor(thrust), torch.tensor(torque)
        )

        fig, ax = plt.subplots()
        traj = clip_to_ground(trajectory.numpy())
        ax.plot(traj[:, 0], traj[:, 1], label="Trajectory")
        ax.axhline(y=0, color='green', linestyle='--')
        ax.scatter([0], [0], color='orange', s=120, zorder=5, label="Landing Pad")
        ax.set_title("Single Rocket Trajectory")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Altitude")
        ax.legend()
        st.pyplot(fig)

        fx, fy = final_state[0].item(), final_state[1].item()
        speed = torch.sqrt(final_state[2] ** 2 + final_state[3] ** 2).item()
        st.write(f"**Final Position:** x={fx:.2f}, y={fy:.2f}")
        st.write(f"**Final Speed:** {speed:.2f} m/s | **Landing Loss:** {compute_loss(final_state).item():.2f}")

# ====================== TAB 2: FLEET SIMULATION ======================
with tab2:
    st.header("Fleet Simulation (Vectorized)")
    st.caption("Simulates every rocket in parallel with torch.vmap.")

    if st.button("Simulate Fleet"):
        # Create random initial states for the fleet
        initial_states = torch.zeros((num_rockets, 7))
        initial_states[:, 0] = torch.randn(num_rockets) * 4
        initial_states[:, 1] = 25 + torch.randn(num_rockets) * 6
        initial_states[:, 4] = torch.randn(num_rockets) * 0.3
        initial_states[:, 6] = 45.0

        # Thrusts span the hover point so some rockets slow down and some fall.
        thrusts = torch.rand(num_rockets) * 120 + 60
        torques = torch.randn(num_rockets) * 2.0

        final_states, trajectories = simulate_fleet(initial_states, thrusts, torques)

        # Plot all trajectories
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(num_rockets):
            traj = clip_to_ground(trajectories[i].numpy())
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.6)
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2)
        ax.scatter([0], [0], color='orange', s=150, zorder=5, label="Landing Pad")
        ax.set_title(f"Fleet Simulation ({num_rockets} Rockets)")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Altitude")
        ax.legend()
        st.pyplot(fig)

        losses = torch.stack([compute_loss(final_states[i]) for i in range(num_rockets)])
        st.write(f"**Average Landing Loss across fleet:** {losses.mean().item():.2f}")

# ====================== TAB 3: GRADIENT OPTIMIZATION ======================
with tab3:
    st.header("Gradient-Based Optimization")
    st.caption("Optimizes a full thrust & torque schedule through the differentiable physics to land softly at the pad.")

    opt_init_x = st.slider("Initial X (Optimization)", -5.0, 5.0, 1.5, key="opt_x")
    opt_init_y = st.slider("Initial Y (Optimization)", 15.0, 40.0, 28.0, key="opt_y")

    opt_initial_state = torch.tensor([opt_init_x, opt_init_y, 0.3, -0.8, 0.15, 0.0, 45.0])

    if st.button("Run Gradient Optimization"):
        with st.spinner("Optimizing thrust & torque schedule..."):
            # Optimize raw parameters and squash them into valid actuator ranges, so the
            # optimizer can never ask for negative or impossibly large thrust.
            thrust_raw = torch.zeros(NUM_STEPS, requires_grad=True)
            torque_raw = torch.zeros(NUM_STEPS, requires_grad=True)

            optimizer = optim.Adam([thrust_raw, torque_raw], lr=0.05)
            losses = []

            for _ in range(450):
                optimizer.zero_grad()
                thrust_seq = torch.sigmoid(thrust_raw) * MAX_THRUST
                torque_seq = torch.tanh(torque_raw) * MAX_TORQUE
                final_state, trajectory = simulate_schedule(opt_initial_state, thrust_seq, torque_seq)
                loss = landing_loss(final_state, trajectory, thrust_seq)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # Final optimized simulation
            with torch.no_grad():
                thrust_seq = torch.sigmoid(thrust_raw) * MAX_THRUST
                torque_seq = torch.tanh(torque_raw) * MAX_TORQUE
                final_opt, trajectory_opt = simulate_schedule(opt_initial_state, thrust_seq, torque_seq)

        distance = torch.sqrt(final_opt[0] ** 2 + final_opt[1] ** 2).item()
        speed = torch.sqrt(final_opt[2] ** 2 + final_opt[3] ** 2).item()
        tilt = final_opt[4].item()

        status, message = landing_verdict(distance, speed, tilt)
        getattr(st, status)(f"Optimization Complete! {message}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Distance to Pad", f"{distance:.2f} m")
        col2.metric("Touchdown Speed", f"{speed:.2f} m/s")
        col3.metric("Final Tilt", f"{tilt:.2f} rad")
        col1.metric("Final Loss", f"{landing_loss(final_opt, trajectory_opt, thrust_seq).item():.2f}")

        # Plot optimized trajectory
        fig, ax = plt.subplots()
        traj = clip_to_ground(trajectory_opt.numpy())
        ax.plot(traj[:, 0], traj[:, 1], color='blue', label="Optimized descent")
        ax.axhline(y=0, color='green', linestyle='--')
        ax.scatter([0], [0], color='orange', s=120, zorder=5, label="Landing Pad")
        ax.set_title("Optimized Landing Trajectory")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Altitude")
        ax.legend()
        st.pyplot(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            # Loss curve
            fig2, ax2 = plt.subplots()
            ax2.plot(losses)
            ax2.set_yscale("log")
            ax2.set_title("Optimization Progress (Loss Curve)")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Loss (log scale)")
            st.pyplot(fig2)
        with col_b:
            # Control schedule
            fig3, ax3 = plt.subplots()
            ax3.plot(thrust_seq.numpy(), label="Thrust")
            ax3.axhline(y=HOVER_THRUST, color='gray', linestyle=':', label="Hover thrust")
            ax3.set_title("Optimized Thrust Schedule")
            ax3.set_xlabel("Time step")
            ax3.set_ylabel("Thrust (N)")
            ax3.legend()
            st.pyplot(fig3)

        # ---- Recommended landing control values ----
        thrust_np = thrust_seq.numpy()
        torque_np = torque_seq.numpy()
        times = np.arange(NUM_STEPS) * DT
        threshold = thrust_np.min() + 0.5 * (thrust_np.max() - thrust_np.min())
        burn_start = int(np.argmax(thrust_np > threshold))

        st.subheader("Recommended landing control values")
        st.caption(
            "A single constant thrust/torque cannot land this rocket — these values are the "
            "time-varying schedule that does. Hold near hover to descend, then burn hard to touch down."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial thrust", f"{thrust_np[0]:.0f} N")
        c2.metric("Coast thrust", f"{np.median(thrust_np[:burn_start]) if burn_start > 0 else thrust_np[0]:.0f} N")
        c3.metric("Peak burn thrust", f"{thrust_np.max():.0f} N")
        c4.metric("Burn starts at", f"{times[burn_start]:.2f} s")

        sample_idx = np.linspace(0, NUM_STEPS - 1, 10).astype(int)
        schedule_table = {
            "time (s)": [f"{times[i]:.2f}" for i in sample_idx],
            "thrust (N)": [f"{thrust_np[i]:.1f}" for i in sample_idx],
            "torque": [f"{torque_np[i]:+.2f}" for i in sample_idx],
        }
        st.table(schedule_table)

        csv = "time_s,thrust_N,torque\n" + "\n".join(
            f"{times[i]:.3f},{thrust_np[i]:.4f},{torque_np[i]:.4f}" for i in range(NUM_STEPS)
        )
        st.download_button("Download full control schedule (CSV)", csv, file_name="landing_schedule.csv")
