import numpy as np
from typing import Tuple
import random
class Grid:

    def __init__(self, size: int = 100, n_obstacles: int = 1000):
        self.size = size
        self.n_obstacles = n_obstacles
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.reset()
        
    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.start = (0, 0)
        self.goal = (self.size-1, self.size-1)
        
        # Place obstacles randomly
        obstacles = random.sample([(i, j) for i in range(self.size) 
                                 for j in range(self.size) 
                                 if (i, j) not in [self.start, self.goal]], 
                                self.n_obstacles)
        for obs in obstacles:
            self.grid[obs] = 1
            
        # Ensure path exists
        while not self._check_path_exists():
            self.remove_random_obstacles(100)
            
        return self.start
    
    def check_path_exists(self):        
        visited = set()
        queue = [self.start]
        while queue:
            current = queue.pop(0)
            if current == self.goal:
                return True
            for action in self.actions:
                next_state = (current[0] + action[0], current[1] + action[1])
                if (self._is_valid_state(next_state) and 
                    next_state not in visited and 
                    self.grid[next_state] == 0):
                    visited.add(next_state)
                    queue.append(next_state)
        return False
    
    def remove_random_obstacles(self, n: int):
        obstacle_positions = np.where(self.grid == 1)
        indices = random.sample(range(len(obstacle_positions[0])), min(n, len(obstacle_positions[0])))
        for idx in indices:
            self.grid[obstacle_positions[0][idx], obstacle_positions[1][idx]] = 0
    
    def is_valid_state(self, state):
        return (0 <= state[0] < self.size and 0 <= state[1] < self.size and self.grid[state] == 0)
    
    def step(self, state, action):
        
        next_state = (state[0] + self.actions[action][0], 
                     state[1] + self.actions[action][1])
        
        if not self._is_valid_state(next_state):
            next_state = state
            reward = -10  # Collision penalty
        elif next_state == self.goal:
            reward = 100  # Goal reward
        else:
            reward = -1  # Step cost

        done = (next_state == self.goal)
        return next_state, reward, done
    # def step(self, state: Tuple[int, int], action: int):
        
        
    #     next_state = (state[0] + self.actions[action][0], 
    #                 state[1] + self.actions[action][1])
    #     done = False
    #     if not self._is_valid_state(next_state):
    #         # Collision with an obstacle or boundary
    #         next_state = state  # Agent stays in the same state
    #         reward = -10  # Collision penalty
    #         done = True  # Episode ends on collision
    #     elif next_state == self.goal:
    #         # Reached the goal
    #         reward = 100  # Goal reward
    #         done = True  # Episode ends when goal is reached
    #     else:
    #         # Regular step
    #         reward = -1  # Step cost
    #         done = False  # Continue the episode
        
    #     return next_state, reward, done

            
        