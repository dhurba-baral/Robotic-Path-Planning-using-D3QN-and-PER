import numpy as np
import torch

# train agent function
def train_agent(agent, env, episodes=500):
    """Train agent and collect metrics"""
    metrics = {
        'episode_rewards': [],
        'avg_q_values': [],
        'collisions': [],
        'path_lengths': [],
        'training_times': []
    }
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_q_values = []
        done = False
        
        while not done:
            action = agent.select_action(state, explore=True)
            
            # Track Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_val = agent.policy_net(state_tensor).max().item()
                episode_q_values.append(q_val)
            
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            episode_reward += reward
        
        agent.decay_epsilon()
        
        # Store metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['avg_q_values'].append(np.mean(episode_q_values) if episode_q_values else 0)
        metrics['collisions'].append(info['collisions'])
        metrics['path_lengths'].append(env.steps)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Collisions: {info['collisions']}")
    
    return metrics