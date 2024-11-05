import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
from typing import Tuple

from grid import Grid

class QLearning:
    def __init__(self, env: Grid, gamma: float = 0.99, alpha: float = 0.1, epsilon: float = 0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.policy = np.zeros((env.size, env.size), dtype=int)
        
    def get_action(self, state):
       
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[state]))
    
    def learn(self, episodes: int = 10000):
       #training
        for _ in tqdm(range(episodes), desc='Q-Learning'):
            state = self.env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(state, action)
                
                # Q-learning update
                self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                    self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]))
                state = next_state
        
        # Extract policy
        for i in range(self.env.size):
            for j in range(self.env.size):
                self.policy[i,j] = np.argmax(self.q_table[(i,j)])