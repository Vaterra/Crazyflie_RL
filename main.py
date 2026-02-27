# First start with "Pretraining" for Evader // Defender

# Chaser is straight going towards the Defender, no RL for Chaser.
import numpy as np

class Pretraining:
    def __init__(
            self,
            space_size=(8, 8, 8),
            knockou_rad = 0.5,
            top_speed = 5.0,
            dt = 0.05,
            max_steps = 200,
            goal_radius = 0.5,
    ):
        self.space_size = space_size
        self.knockou_rad = knockou_rad
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
        
        self.evader_pos = (
            rng.uniform(0, self.goal_radius*2),
            rng.uniform(self.space_size[1]-3*self.goal_radius, self.space_size[1]-self.goal_radius),
            rng.uniform(0, self.space_size[2])
        )
        self.evader_vel = (0.0, 0.0, 0.0)
        self.chaser_pos = (
            rng.uniform(0, self.space_size[0]),
            rng.uniform(0, self.space_size[1]),
            rng.uniform(0, self.space_size[2])
        )
        self.chaser_vel = (0.0, 0.0, 0.0)
        self.goal = (
            rng.uniform(self.space_size[0]/2 - self.goal_radius, self.space_size[0]/2 + self.goal_radius),
            rng.uniform(self.space_size[1]-3*self.goal_radius, self.space_size[1]-self.goal_radius),
            rng.uniform(self.space_size[2]/2-self.goal_radius, self.space_size[2]/2+self.goal_radius)
        )
        return self.get_state()
    
    def get_state(self):
        return {
            'evader_pos': self.evader_pos.copy(),
            'evader_vel': self.evader_vel.copy(),
            'chaser_pos': self.chaser_pos.copy(),
            'chaser_vel': self.chaser_vel.copy(),
            'goal': self.goal.copy()
        }

    def chaser_straight_2_defender(chaser_pos, evader_pos, top_speed):
        # Move towards the defender
        dx = evader_pos[0] - chaser_pos[0]
        dy = evader_pos[1] - chaser_pos[1]
        dz = evader_pos[2] - chaser_pos[2]
        distance = (dx**2 + dy**2 + dz**2)**0.5
        if distance > 0:
            vx = dx / distance * top_speed
            vy = dy / distance * top_speed
            vz = dz / distance * top_speed
            vel = (vx, vy, vz)
            return vel
        return 0, 0, 0

    def euler_step(position, velocity, dt):
        new_position = (
            position[0] + velocity[0] * dt,
            position[1] + velocity[1] * dt,
            position[2] + velocity[2] * dt
        )
        return new_position