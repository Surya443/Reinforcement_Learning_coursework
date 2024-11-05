import numpy as np
from grid import Grid

class ValueIteration:
    """Value Iteration (Dynamic Programming) algorithm"""
    def __init__(self, env: Grid, gamma: float = 0.99):
        self.env = env
        self.gamma = gamma
        self.values = np.zeros((env.size, env.size))
        self.policy = np.zeros((env.size, env.size), dtype=int)
        
    def solve(self, theta: float = 1e-6, max_iterations: int = 1000):
        
        for iteration in range(max_iterations):
            delta = 0
            for i in range(self.env.size):
                for j in range(self.env.size):
                    if self.env.grid[i,j] == 1:
                        continue
                    
                    v = self.values[i,j]
                    values = []
                    for action in range(4):
                        next_state, reward, _ = self.env.step((i,j), action)
                        values.append(reward + self.gamma * self.values[next_state])
                    
                    self.values[i,j] = max(values)
                    self.policy[i,j] = np.argmax(values)
                    delta = max(delta, abs(v - self.values[i,j]))
            
            if delta < theta:
                return iteration + 1
        return max_iterations