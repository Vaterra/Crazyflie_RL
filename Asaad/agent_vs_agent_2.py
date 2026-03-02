#!/usr/bin/env python3
"""
Agent-vs-agent demo: Evader PPO vs Chaser PPO in the Pretraining environment.
"""

import sys
import types
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection import)
import os


# ----------------------------------------------------------------------
# NumPy _core compatibility hack
# (needed if your saved SB3 models reference numpy._core.* when unpickling)
# ----------------------------------------------------------------------
import numpy.core.numeric as _real_numeric
import numpy.core.multiarray as _real_multiarray

core_pkg = types.ModuleType("numpy._core")
core_pkg.__path__ = []  # mark as a package

numeric_mod = types.ModuleType("numpy._core.numeric")
numeric_mod.__dict__.update(_real_numeric.__dict__)

multiarray_mod = types.ModuleType("numpy._core.multiarray")
multiarray_mod.__dict__.update(_real_multiarray.__dict__)

sys.modules["numpy._core"] = core_pkg
sys.modules["numpy._core.numeric"] = numeric_mod
sys.modules["numpy._core.multiarray"] = multiarray_mod

# ----------------------------------------------------------------------
# Stable-Baselines3 + environment
# ----------------------------------------------------------------------
from stable_baselines3 import PPO
from main_sim import Pretraining


# === CONFIG ===
BASE_DIR = os.path.dirname(__file__)  # directory of agent_vs_agent_2.py

EVADER_MODEL_PATH = "evader_pretrain_ppo.zip"   # <- change if needed
CHASER_MODEL_PATH = os.path.join(BASE_DIR, "Assad_def_stage_2")
N_EPISODES = 10
ACTION_SCALE = 5.0  # same scaling you used during training (action * 10)


# ----------------------------------------------------------------------
# Observation builders
# ----------------------------------------------------------------------
def build_evader_obs(state: dict) -> np.ndarray:
    """
    Observation format used for the EVADER PPO (18-dim):
      [evader_pos(3), evader_vel(3),
       chaser_pos(3), chaser_vel(3),
       relative_pos(3),  # chaser_pos - evader_pos
       goal_pos(3)]
    """
    rel = state["chaser_pos"] - state["evader_pos"]
    obs = np.concatenate([
        state["evader_pos"],
        state["evader_vel"],
        state["chaser_pos"],
        state["chaser_vel"],
        rel,
        state["goal"],
    ]).astype(np.float32)
    # Sanity check:
    # assert obs.shape == (18,)
    return obs


def build_chaser_obs(state: dict, sim: Pretraining) -> np.ndarray:
    """
    Build a 15-dim observation compatible with the IntrusionImpulseSingleEnv
    'defender' (chaser) style observation:

      [t_n,
       dp_n(3),
       d_n,
       dv_n(3),
       g_n(3),
       dg_n,
       v_n(3)]

    where:
      - defender = chaser
      - intruder = evader
    """
    # Treat chaser as defender, evader as intruder
    pD = state["chaser_pos"].astype(np.float32)
    vD = state["chaser_vel"].astype(np.float32)
    pI = state["evader_pos"].astype(np.float32)
    vI = state["evader_vel"].astype(np.float32)
    goal = state["goal"].astype(np.float32)

    room_min = np.zeros(3, dtype=np.float32)
    room_max = np.array(sim.space_size, dtype=np.float32)
    room_span = room_max - room_min
    room_diag = float(np.linalg.norm(room_span))
    v_scale = float(sim.top_speed)

    # defender-style features
    dp = (pI - pD)              # intruder - defender
    dv = (vI - vD)              # intruder - defender
    d = float(np.linalg.norm(dp))
    g = (goal - pD)
    dg = float(np.linalg.norm(g))

    dp_n = np.clip(dp / room_span, -1, 1)
    dv_n = np.clip(dv / v_scale, -1, 1)
    d_n = np.clip(d / room_diag, 0, 1) * 2 - 1
    g_n = np.clip(g / room_span, -1, 1)
    dg_n = np.clip(dg / room_diag, 0, 1) * 2 - 1
    v_n = np.clip(vD / v_scale, -1, 1)

    # normalized time in [-1, 1]
    t_n = 2.0 * (state["step_count"] / sim.max_steps) - 1.0

    obs = np.concatenate([
        np.array([t_n], dtype=np.float32),
        dp_n,
        np.array([d_n], dtype=np.float32),
        dv_n,
        g_n,
        np.array([dg_n], dtype=np.float32),
        v_n
    ]).astype(np.float32)

    # Sanity check:
    # assert obs.shape == (15,)
    return obs


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_last_episode(evader_traj, chaser_traj, goal, space_size, goal_radius):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Evader trajectory
    ax.plot(
        evader_traj[:, 0],
        evader_traj[:, 1],
        evader_traj[:, 2],
        label="Evader",
        linewidth=2,
    )

    # Chaser trajectory
    ax.plot(
        chaser_traj[:, 0],
        chaser_traj[:, 1],
        chaser_traj[:, 2],
        label="Chaser",
        linewidth=2,
    )

    # Start positions
    ax.scatter(
        evader_traj[0, 0],
        evader_traj[0, 1],
        evader_traj[0, 2],
        s=80,
        marker="o",
        label="Evader Start",
    )
    ax.scatter(
        chaser_traj[0, 0],
        chaser_traj[0, 1],
        chaser_traj[0, 2],
        s=80,
        marker="o",
        label="Chaser Start",
    )

    # Goal point
    ax.scatter(
        goal[0],
        goal[1],
        goal[2],
        s=150,
        marker="X",
        label="Goal",
    )

    # Plot goal sphere for visualization
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x = goal[0] + goal_radius * np.cos(u) * np.sin(v)
    y = goal[1] + goal_radius * np.sin(u) * np.sin(v)
    z = goal[2] + goal_radius * np.cos(v)
    ax.plot_wireframe(x, y, z, alpha=0.2, linewidth=0.5)

    # Bounds (the "map" cube)
    ax.set_xlim(0, space_size[0])
    ax.set_ylim(0, space_size[1])
    ax.set_zlim(0, space_size[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Evader vs Chaser – Episode Trajectories")
    ax.legend()

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main demo
# ----------------------------------------------------------------------
def run_demo():
    # Load models (force CPU to avoid the PPO GPU warning)
    evader_model = PPO.load(EVADER_MODEL_PATH)
    chaser_model = PPO.load(CHASER_MODEL_PATH)

    # Shared simulator
    sim = Pretraining()
    space_size = sim.space_size
    goal_radius = sim.goal_radius

    # Stats
    term_counts = {
        "evader_reached_goal": 0,
        "captured": 0,
        "evader_out": 0,
        "chaser_out": 0,
        "timeout": 0,
        "unknown": 0,
    }
    episode_lengths = []

    # For prioritized plotting
    best_goal_episode = None
    best_capture_episode = None
    last_episode_data = None

    for ep in range(N_EPISODES):
        state = sim.reset()
        done = False

        evader_traj = [state["evader_pos"].copy()]
        chaser_traj = [state["chaser_pos"].copy()]
        goal = state["goal"].copy()

        step_count = 0
        info = {}

        while not done:
            # Build obs for each agent separately
            evader_obs = build_evader_obs(state)         # 18-dim
            chaser_obs = build_chaser_obs(state, sim)    # 15-dim

            # Evader action
            evader_action, _ = evader_model.predict(
                evader_obs,
                deterministic=True
            )

            # Chaser action
            chaser_action, _ = chaser_model.predict(
                chaser_obs,
                deterministic=True
            )

            # Scale actions to match training
            evader_acc = evader_action * ACTION_SCALE
            chaser_acc = chaser_action * ACTION_SCALE

            # Step sim
            state, done, info = sim.step(
                evader_action=evader_acc,
                chaser_action=chaser_acc,
            )

            evader_traj.append(state["evader_pos"].copy())
            chaser_traj.append(state["chaser_pos"].copy())

            step_count += 1

        # Episode done – update stats
        if info.get("evader_reached_goal"):
            term_counts["evader_reached_goal"] += 1
            print(f"Episode {ep+1}: Evader reached the goal 🏁")
        elif info.get("captured"):
            term_counts["captured"] += 1
            print(f"Episode {ep+1}: Evader was captured 💥")
        elif info.get("evader_out"):
            term_counts["evader_out"] += 1
            print(f"Episode {ep+1}: Evader went out of bounds")
        elif info.get("chaser_out"):
            term_counts["chaser_out"] += 1
            print(f"Episode {ep+1}: Chaser went out of bounds")
        elif info.get("timeout"):
            term_counts["timeout"] += 1
            print(f"Episode {ep+1}: Timeout")
        else:
            term_counts["unknown"] += 1
            print(f"Episode {ep+1}: Termination reason unknown 🤔")

        episode_lengths.append(step_count)

        episode_data = (
            np.array(evader_traj),
            np.array(chaser_traj),
            goal.copy(),
            info.copy(),
        )

        # Highest priority: evader reaches goal
        if info.get("evader_reached_goal"):
            best_goal_episode = episode_data

        # Second priority: captured
        elif info.get("captured"):
            best_capture_episode = episode_data

        # Always store last episode as fallback
        last_episode_data = episode_data

    # === Print summary stats ===
    print("\n=== Summary over", N_EPISODES, "episodes ===")
    for key, val in term_counts.items():
        if val > 0:
            pct = 100.0 * val / N_EPISODES
            print(f"{key:18s}: {val:4d} episodes ({pct:5.1f}%)")

    mean_len = np.mean(episode_lengths) if episode_lengths else 0.0
    print(f"Mean episode length: {mean_len:.2f} steps")

    # Decide which episode to plot (priority: goal > capture > last)
    if best_goal_episode is not None:
        print("\nPlotting episode where evader reached goal 🏁")
        evader_traj, chaser_traj, goal, last_info = best_goal_episode

    elif best_capture_episode is not None:
        print("\nPlotting episode where evader was captured 💥")
        evader_traj, chaser_traj, goal, last_info = best_capture_episode

    else:
        print("\nPlotting last episode (no goal/capture occurred)")
        evader_traj, chaser_traj, goal, last_info = last_episode_data

    plot_last_episode(
        evader_traj,
        chaser_traj,
        goal,
        space_size,
        goal_radius,
    )


if __name__ == "__main__":
    run_demo()