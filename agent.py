import numpy as np
import torch
from models import DQN, DuelingDQN
from PER import PrioritizedReplayBuffer, ReplayBuffer
from collections import namedtuple
import random
import os

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Agent
class Agent:
    """RL Agent supporting DQN, DDQN, D3QN variants"""
    def __init__(self, state_size, action_size, model_type='DQN', use_per=False):
        self.state_size = state_size
        self.action_size = action_size
        self.model_type = model_type
        self.use_per = use_per
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update = 10

        # Networks - Choose architecture based on model type
        if 'D3QN' in model_type:
            # D3QN uses Dueling architecture
            self.policy_net = DuelingDQN(state_size, action_size)
            self.target_net = DuelingDQN(state_size, action_size)
        else:
            # DQN and DDQN use standard DQN architecture
            self.policy_net = DQN(state_size, action_size)
            self.target_net = DQN(state_size, action_size)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Replay Buffer
        if use_per:
            self.memory = PrioritizedReplayBuffer(10000)
        else:
            self.memory = ReplayBuffer(10000)

        self.steps = 0

    # action selection
    def select_action(self, state, explore=True):
        """Epsilon-greedy action selection"""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train(self):
        """Train the agent"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample from memory
        if self.use_per:
            experiences, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights)
        else:
            experiences = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size)
        
        # Prepare batch
        batch = Experience(*zip(*experiences))
        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(batch.done)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions).squeeze()
        
        # Target Q values
        with torch.no_grad():
            if self.model_type == 'DQN':
                # Standard DQN: use target network max Q-value
                next_q_values = self.target_net(next_states).max(1)[0]
            elif 'DDQN' in self.model_type or 'D3QN' in self.model_type:
                # Double DQN / D3QN: use policy net to select action, target net to evaluate
                next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        if self.use_per:
            self.memory.update_priorities(indices, td_errors.detach().numpy())
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        experience = Experience(state, action, reward, next_state, done)
        if self.use_per:
            self.memory.add(experience)
        else:
            self.memory.add(experience)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):
        """Save the model"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'model_type': self.model_type,
            'use_per': self.use_per
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")
    
        