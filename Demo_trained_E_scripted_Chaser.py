import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO
from Evader_env import EvaderPretrainEnv


def plot_single_rollout(model_path="evader_pretrain_ppo.zip"):
    env = EvaderPretrainEnv()
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

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

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

    state = env.sim.get_state()
    evader_traj.append(state["evader_pos"].copy())
    chaser_traj.append(state["chaser_pos"].copy())

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

    plt.show()


if __name__ == "__main__":
    plot_single_rollout()