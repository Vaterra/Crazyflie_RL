import numpy as np
def chaser_straight_2_evader(state, max_acc: float):
    # Move towards the defender

    chaser_pos = state["chaser_pos"]
    evader_pos = state["evader_pos"]

    direction = evader_pos - chaser_pos
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        # Same position or extremely close: no clear direction
        return np.zeros(3, dtype=np.float32)
    
    direction = direction / distance  # unit vector
    acc = direction * max_acc

    return acc.astype(np.float32)

def evader_straight_2_goal(state, max_acc: float):
    # Move towards the defender

    evader_pos = state["evader_pos"]
    goal_pos = state["goal"]

    direction = evader_pos - goal_pos
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        # Same position or extremely close: no clear direction
        return np.zeros(3, dtype=np.float32)
    
    direction = direction / distance  # unit vector
    acc = direction * max_acc

    return acc.astype(np.float32)