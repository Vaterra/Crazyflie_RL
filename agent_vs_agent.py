import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection import)

from stable_baselines3 import PPO
from main_sim import Pretraining

# === CONFIG ===
EVADER_MODEL_PATH = "evader_pretrain_ppo.zip"      # <- change if needed
CHASER_MODEL_PATH = "chaser_pretrain_ppo.zip"      # <- change if needed
N_EPISODES = 50
ACTION_SCALE = 10.0   # same scaling you used during training (action * 10)


def build_obs(state):
    """
    Observation format used in both your EvaderPretrainEnv and ChaserPretrainEnv:

    [evader_pos(3), evader_vel(3),
     chaser_pos(3), chaser_vel(3),
     relative_pos(3),   # chaser_pos - evader_pos
     goal_pos(3)]
    """
    rel = state["chaser_pos"] - state["evader_pos"]
    return np.concatenate([
        state["evader_pos"],
        state["evader_vel"],
        state["chaser_pos"],
        state["chaser_vel"],
        rel,
        state["goal"],
    ]).astype(np.float32)


def run_demo():
    # Load models
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

    # For plotting last episode
    last_evader_traj = None
    last_chaser_traj = None
    last_goal = None
    last_info = None

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
            # Build obs for both agents
            obs = build_obs(state)

            # Evader action
            evader_action, _ = evader_model.predict(
                obs,
                deterministic=True
            )
            # Chaser action
            chaser_action, _ = chaser_model.predict(
                obs,
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

        #print(f"Episode {ep+1}/{N_EPISODES} finished after {step_count} steps.")

    # === Print summary stats ===
    print("\n=== Summary over", N_EPISODES, "episodes ===")
    for key, val in term_counts.items():
        if val > 0:
            pct = 100.0 * val / N_EPISODES
            print(f"{key:18s}: {val:4d} episodes ({pct:5.1f}%)")

    mean_len = np.mean(episode_lengths) if episode_lengths else 0.0
    print(f"Mean episode length: {mean_len:.2f} steps")

    if last_info is not None:
        print("\nLast episode termination info:")
        for k, v in last_info.items():
            print(f"  {k}: {v}")

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
    ax.set_title("Evader vs Chaser – Last Episode Trajectories")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()