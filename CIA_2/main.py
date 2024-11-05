import numpy as np
import matplotlib.pyplot as plt
import time

from grid import Grid
from policy import Policy
from sarsa import SARSA
from qlearning import QLearning
from dp import ValueIteration

def compare():   
    
    env = Grid(size=50, n_obstacles=100)
    print("\nGrid Initilized")

    # DP
    print("\nRunning Dynamic Programming - Value Iteration...")
    vi = ValueIteration(env)
    vi_start = time.time()
    vi_iterations = vi.solve()
    vi_time = time.time() - vi_start
    vi_reward = Policy.evaluate_policy(env, vi.policy)
    
    # Q-Learning
    print("\nRunning Q-Learning...")
    ql = QLearning(env)
    ql_start = time.time()
    ql.learn()
    ql_time = time.time() - ql_start
    ql_reward = Policy.evaluate_policy(env, ql.policy)
    
    # SARSA
    print("\nRunning SARSA...")
    sarsa = SARSA(env)
    sarsa_start = time.time()
    sarsa.learn()
    sarsa_time = time.time() - sarsa_start
    sarsa_reward = Policy.evaluate_policy(env, sarsa.policy)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes[0,0].imshow(env.grid, cmap='binary')
    axes[0,0].set_title('Environment\n(Black: Obstacles)')
    
    im = axes[0,1].imshow(vi.policy, cmap='viridis')
    axes[0,1].set_title(f'Value Iteration Policy\nTime: {vi_time:.2f}s\nReward: {vi_reward:.1f}')
    

    axes[1,0].imshow(ql.policy, cmap='viridis')
    axes[1,0].set_title(f'Q-Learning Policy\nTime: {ql_time:.2f}s\nReward: {ql_reward:.1f}')
    
    axes[1,1].imshow(sarsa.policy, cmap='viridis')
    axes[1,1].set_title(f'SARSA Policy\nTime: {sarsa_time:.2f}s\nReward: {sarsa_reward:.1f}')
    
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.show()
    
    
    print("\nResults:")
    print(f"\nValue Iteration:")
    print(f"Time: {vi_time:.2f} seconds")
    print(f"Iterations: {vi_iterations}")
    print(f"Average Reward: {vi_reward:.1f}")
    
    print(f"\nQ-Learning:")
    print(f"Time: {ql_time:.2f} seconds")
    print(f"Average Reward: {ql_reward:.1f}")
    
    print(f"\nSARSA:")
    print(f"Time: {sarsa_time:.2f} seconds")
    print(f"Average Reward: {sarsa_reward:.1f}")

if __name__ == "__main__":
    compare() 