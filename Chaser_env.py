import numpy as np
import gymnasium as gym
from gymnasium import spaces

from main_sim import Pretraining
from Pretraining_scripts import evader_straight_2_goal

class ChaserPretrainEnv(gym.Env):
    """
    Agent: Chaser
    Opponent: Scripted evader (straight to goal)
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
        # Action: chaser acc in x,y,z
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
        state = self.sim.reset()
        obs = self._get_obs(state)
        return obs, {}
    
    def step(self, action):
        # Scripted evader
        evader_action = evader_straight_2_goal(self.sim.get_state(), max_acc=self.max_acc*10)

        state, done, info = self.sim.step(
            evader_action=evader_action,
            chaser_action=action*10,
        )

        obs = self._get_obs(state)

        reward = 0.0
        reward += 0.01 # Reward for each step survived...
        reward -= 0.1 * np.linalg.norm(state["chaser_pos"] - state["evader_pos"])
        reward -= 0.01 * float(np.linalg.norm(action) ** 2) #

        if info["captured"]:
            reward += 100.0
        if info["evader_reached_goal"]:
            reward -= 100.0
        if info["chaser_out"]:
            reward -= 100.0

        terminated = (
            info["captured"] or
            info["evader_reached_goal"] or
            info["evader_out"] or
            info["chaser_out"]
        )
        truncated = info["timeout"]

        return obs, reward, terminated, truncated, info