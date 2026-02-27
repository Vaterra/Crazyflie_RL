# First start with "Pretraining" for Evader // Defender

# Chaser is straight going towards the Defender, no RL for Chaser.
import numpy as np

class Pretraining:
    def __init__(
            self,
            space_size=(8, 8, 8),
            knockout_rad = 0.5,
            top_speed = 5.0,
            dt = 0.05,
            max_steps = 200,
            goal_radius = 0.5,
    ):
        self.space_size = space_size
        self.knockout_rad = knockout_rad
        self.top_speed = top_speed
        self.dt = dt
        self.max_steps = max_steps
        self.goal_radius = goal_radius

        self.evader_pos = None
        self.evader_vel = None
        self.chaser_pos = None
        self.chaser_vel = None
        self.goal = None
        self.step_count = 0
        
    def reset(self, rng = None):
        if rng is None:
            rng = np.random.default_rng()

        self.step_count = 0
        
        self.evader_pos = np.array([
            rng.uniform(self.space_size[0]/2-self.goal_radius, self.space_size[0]/2+self.goal_radius),
            rng.uniform(self.goal_radius*2, self.goal_radius*4),
            rng.uniform(self.goal_radius*2, self.space_size[2]-self.goal_radius*2)
        ])
        self.evader_vel = np.array([0.0, 0.0, 0.0])
        self.chaser_pos = np.array([
            rng.uniform(self.space_size[0]/2-self.goal_radius, self.space_size[0]/2+self.goal_radius),
            rng.uniform(self.space_size[1]/2-self.goal_radius, self.space_size[1]/2+self.goal_radius),
            rng.uniform(0, self.space_size[2])
        ])
        self.chaser_vel = np.array([0.0, 0.0, 0.0])
        self.goal = np.array([  
            rng.uniform(self.space_size[0]/2-self.goal_radius, self.space_size[0]/2+self.goal_radius),
            rng.uniform(self.space_size[1]-3*self.goal_radius, self.space_size[1]-self.goal_radius),
            rng.uniform(0, self.space_size[2])
        ])
        return self.get_state()
    
    def get_state(self):
        return {
            'evader_pos': self.evader_pos.copy(),
            'evader_vel': self.evader_vel.copy(),
            'chaser_pos': self.chaser_pos.copy(),
            'chaser_vel': self.chaser_vel.copy(),
            'goal': self.goal.copy(),
            "step_count": self.step_count
        }


    def euler_step(self, position, velocity, acc):
        velocity =  velocity + acc * self.dt

        speed = np.linalg.norm(velocity)
        if speed > self.top_speed:
            velocity = velocity / speed * self.top_speed


        new_position = position + velocity * self.dt

        return new_position, velocity
    
    def _distance_evader_chaser(self):
        return float(np.linalg.norm(self.chaser_pos - self.evader_pos))
    
    def _distance_evader_goal(self):
        return float(np.linalg.norm(self.goal - self.evader_pos))
    
    def _in_bounds(self, pos):
        return np.all(pos >= 0.0) and np.all(pos <= self.space_size)

    def step(self, evader_action, chaser_action):
        self.step_count += 1

        evader_action = np.asarray(evader_action, dtype=np.float32)
        chaser_action = np.asarray(chaser_action, dtype=np.float32)

        self.evader_pos, self.evader_vel = self.euler_step(self.evader_pos, self.evader_vel, evader_action)
        self.chaser_pos, self.chaser_vel = self.euler_step(self.chaser_pos, self.chaser_vel, chaser_action)

        dist = self._distance_evader_chaser()
        goal_dist = self._distance_evader_goal()

        captured = dist <= self.knockout_rad
        reached_goal = goal_dist <= self.goal_radius
        evader_out = not self._in_bounds(self.evader_pos)
        chaser_out = not self._in_bounds(self.chaser_pos)
        timeout = self.step_count >= self.max_steps


        done = captured or reached_goal or evader_out or chaser_out or timeout

        info = {
            "captured": captured,
            "evader_reached_goal": reached_goal,
            "evader_out": evader_out,
            "chaser_out": chaser_out,
            "timeout": timeout,
            "distance": dist,
            "goal_distance": goal_dist,
        }
        return self.get_state(), done, info