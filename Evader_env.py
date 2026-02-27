import numpy as np
import gymnasium as gym
from gymnasium import spaces

from main_sim import Pretraining
from Pretraining_scripts import chaser_straight_2_evader


class EvaderPretrainEnv(gym.Env):
    """
    Agent: Evader
    Opponent: Scripted chaser (straight to evader)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.termination_stats = {
            "goal": 0,
            "captured": 0,
            "evader_out": 0,
            "chaser_out": 0,
            "timeout": 0,
        }

        self.sim = Pretraining()

        self.max_acc = 1.0
        # Action: evader acc in x,y,z
        self.action_space = spaces.Box(
            low=- self.max_acc,
            high=self.max_acc,
            shape=(3,),
            dtype=np.float32,
        )
        # Observation:
        # [evader_pos(3), evader_vel(3),
        #  chaser_pos(3), chaser_vel(3),
        #  relative_pos(3),
        #  goal_pos(3)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )
    def _get_obs(self, state):
        rel = state["chaser_pos"] - state["evader_pos"]
        return np.concatenate([
            state["evader_pos"],
            state["evader_vel"],
            state["chaser_pos"],
            state["chaser_vel"],
            rel,
            state["goal"],
        ]).astype(np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        state = self.sim.reset()
        dist_goal = np.linalg.norm(state["goal"] - state["evader_pos"])
        self.prev_goal_dist = dist_goal
        obs = self._get_obs(state)

        return obs, {}

    def step(self, action):
        # Scripted chaser
        chaser_action = chaser_straight_2_evader(self.sim.get_state(), max_acc=self.max_acc*10)

        state, done, info = self.sim.step(
            evader_action=action*10,
            chaser_action=chaser_action,
        )

        dist_goal = info["goal_distance"]
        dist_chaser = info["distance"]

        progress = self.prev_goal_dist - dist_goal
        self.prev_goal_dist = dist_goal

        # Reward design (evader wants goal + survival)
        reward = 0.0
        reward += 1 * progress                   # main drive: get closer each step
        reward -= 0.005  
        #reward -= 0.01 * max(0.0, 5.0 - dist_chaser) # avoid close chaser
        #reward -= 0.001 * float(np.linalg.norm(action) ** 2)

        if info["evader_reached_goal"]:
            reward += 100.0
        if info["captured"]:
            reward -= 100.0
        if info["evader_out"]:
            reward -= 100.0
        if info["timeout"]:
            reward -= 100.0
        obs = self._get_obs(state)

        terminated = (
            info["captured"]
            or info["evader_reached_goal"]
            or info["evader_out"]
            #or info["chaser_out"]
        )

        truncated = info["timeout"]

        if terminated or truncated:
            if info["evader_reached_goal"]:
                self.termination_stats["goal"] += 1
            elif info["captured"]:
                self.termination_stats["captured"] += 1
            elif info["evader_out"]:
                self.termination_stats["evader_out"] += 1
            elif info["chaser_out"]:
                self.termination_stats["chaser_out"] += 1
            elif info["timeout"]:
                self.termination_stats["timeout"] += 1

        return obs, reward, terminated, truncated, info