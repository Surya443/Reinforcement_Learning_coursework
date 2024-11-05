import numpy as np

class Policy:
    def evaluate_policy(env, policy, n_episodes=100):
        
        total_reward = 0
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                
                if isinstance(policy, np.ndarray):  
                    action = policy[state[0], state[1]]
                else:  
                    action = np.argmax(policy[state])
                
                state, reward, done = env.step(state, action)
                episode_reward += reward
            total_reward += episode_reward
        return total_reward / n_episodes