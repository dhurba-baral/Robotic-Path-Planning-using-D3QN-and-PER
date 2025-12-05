import numpy as np
import torch

def test_agent(agent, env, episodes=100):
    """Test agent on test environment (no training, no exploration)"""
    print(f"\nTesting agent on {'Dynamic' if env.dynamic else 'Static'} maze...")

    metrics = {
        'episode_rewards': [],
        'avg_q_values': [],
        'collisions': [],
        'path_lengths': []
    }

    # Set epsilon to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_q_values = []
        done = False

        while not done:
            action = agent.select_action(state, explore=False)

            # Track Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_val = agent.policy_net(state_tensor).max().item()
                episode_q_values.append(q_val)

            next_state, reward, done, info = env.step(action)

            state = next_state
            episode_reward += reward

        # Store metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['avg_q_values'].append(np.mean(episode_q_values) if episode_q_values else 0)
        metrics['collisions'].append(info['collisions'])
        metrics['path_lengths'].append(env.steps)

    # Restore original epsilon
    agent.epsilon = original_epsilon

    print(f"Test completed: Avg Reward: {np.mean(metrics['episode_rewards']):.2f}, "
          f"Avg Collisions: {np.mean(metrics['collisions']):.1f}, "
          f"Avg Path Length: {np.mean(metrics['path_lengths']):.1f}")

    return metrics