from grid import Grid

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
from typing import Tuple, List

class SARSA:
    """SARSA algorithm"""
    def __init__(self, env, gamma: float = 0.99, alpha: float = 0.1, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.policy = np.zeros((env.size, env.size), dtype=int)
        
    def get_action(self, state: Tuple[int, int]) -> int:
        
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[state]))
    
    def learn(self, episodes: int = 10000):
       
        for _ in tqdm(range(episodes), desc='SARSA'):
            state = self.env.reset()
            action = self.get_action(state)
            done = False
            
            while not done:
                next_state, reward, done = self.env.step(state, action)
                next_action = self.get_action(next_state)
                
                
                self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                    self.alpha * (reward + self.gamma * self.q_table[next_state][next_action])
                
                state = next_state
                action = next_action
        
        # Extract policy
        for i in range(self.env.size):
            for j in range(self.env.size):
                self.policy[i,j] = np.argmax(self.q_table[(i,j)])