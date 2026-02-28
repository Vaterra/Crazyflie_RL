import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO
from Evader_env import EvaderPretrainEnv
from Chaser_env import ChaserPretrainEnv


import numpy as np
import csv
from collections import Counter


def plot_single_rollout(model_path):
    if model_path == "evader_pretrain_ppo.zip":
        print("=========================================")
        print("Loaded trained evader")
        print("=========================================")
        env = EvaderPretrainEnv()
    else:
        print("=========================================")
        print("Loaded trained chaser")
        print("=========================================")
        env = ChaserPretrainEnv()
    space_size = env.sim.space_size
    model = PPO.load(model_path)

    obs, info = env.reset()

    evader_traj = []
    chaser_traj = []

    state = env.sim.get_state()
    evader_traj.append(state["evader_pos"].copy())
    chaser_traj.append(state["chaser_pos"].copy())
    goal = state["goal"].copy()

    done = False
    prev_pos = state["evader_pos"].copy()

    step_idx = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = env.sim.get_state()
        current_pos = state["evader_pos"]

        # displacement this step
        step_distance = np.linalg.norm(current_pos - prev_pos)


        # 🔹 Append positions every step
        evader_traj.append(state["evader_pos"].copy())
        chaser_traj.append(state["chaser_pos"].copy())

        prev_pos = current_pos.copy()
        step_idx += 1
        

    # After loop finishes:
    print("\n=== Episode Terminated ===")

    if info["evader_reached_goal"]:
        print("Reason: Evader reached the goal")

    elif info["captured"]:
        print("Reason: Evader was captured by the chaser")

    elif info["evader_out"]:
        print("Reason: Evader went out of bounds")

    elif info["chaser_out"]:
        print("Reason: Chaser went out of bounds")

    elif info["timeout"]:
        print("Reason: Timeout")

    else:
        print("Reason: Unknown")

    #evader_traj.append(state["evader_pos"].copy())
    #chaser_traj.append(state["chaser_pos"].copy())

    evader_traj = np.array(evader_traj)
    chaser_traj = np.array(chaser_traj)

    # --- Plot ---
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
        s=100,
        marker="o",
        label="Evader Start",
    )

    ax.scatter(
        chaser_traj[0, 0],
        chaser_traj[0, 1],
        chaser_traj[0, 2],
        s=100,
        marker="o",
        label="Chaser Start",
    )

    # Goal
    ax.scatter(
        goal[0],
        goal[1],
        goal[2],
        s=150,
        marker="X",
        label="Goal",
    )

    ax.set_xlim(0, space_size[0])
    ax.set_ylim(0, space_size[1])
    ax.set_zlim(0, space_size[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Evader vs Scripted Chaser Trajectory")
    ax.legend()

        # --- Goal sphere (visualizing goal_radius) ---
    goal_radius = env.sim.goal_radius  # or env.sim.goal_rad

    # Create a sphere around the goal
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = goal[0] + goal_radius * np.cos(u) * np.sin(v)
    y = goal[1] + goal_radius * np.sin(u) * np.sin(v)
    z = goal[2] + goal_radius * np.cos(v)

    ax.plot_wireframe(x, y, z, alpha=0.3, linewidth=0.5)

    plt.show()

def evaluate_policy(
    model_path,
    n_episodes=10,
    csv_path=None,      # e.g. "eval_results.csv"
):
    if model_path == "evader_pretrain_ppo.zip":
        env = EvaderPretrainEnv()
    else:
        env = ChaserPretrainEnv()
    model = PPO.load(model_path)

    # Count how often each termination reason occurs
    reason_counter = Counter()

    # Optional: store per-episode results
    records = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        step_count = 0
        total_reward = 0.0

        last_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1
            last_info = info

        # Determine termination reason (same logic as in your plotting code)
        if last_info is None:
            reason = "unknown"
        elif last_info.get("evader_reached_goal", False):
            reason = "evader_reached_goal"
        elif last_info.get("captured", False):
            reason = "captured"
        elif last_info.get("evader_out", False):
            reason = "evader_out"
        elif last_info.get("chaser_out", False):
            reason = "chaser_out"
        elif last_info.get("timeout", False):
            reason = "timeout"
        else:
            reason = "unknown"

        reason_counter[reason] += 1

        records.append({
            "episode": ep,
            "reason": reason,
            "steps": step_count,
            "total_reward": float(total_reward),
            "distance": float(last_info.get("distance", np.nan)) if last_info else np.nan,
            "goal_distance": float(last_info.get("goal_distance", np.nan)) if last_info else np.nan,
        })

    # Print summary
    print(f"\n=== Evaluation over {n_episodes} episodes ===")
    for reason, count in reason_counter.items():
        pct = 100.0 * count / n_episodes
        print(f"{reason:20s}: {count:4d} episodes ({pct:5.1f}%)")

    # Optionally save to CSV for later analysis
    if csv_path is not None:
        fieldnames = ["episode", "reason", "steps", "total_reward", "distance", "goal_distance"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow(row)
        print(f"\nSaved detailed results to {csv_path}")

if __name__ == "__main__":
    modelpath = "chaser_pretrain_ppo.zip" # "evader_pretrain_ppo.zip"
    n = 1000
    evaluate_policy(modelpath, n_episodes=n)
    plot_single_rollout(modelpath)