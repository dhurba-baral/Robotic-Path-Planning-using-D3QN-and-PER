import torch.nn as nn

class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DuelingDQN(nn.Module):
    """Dueling DQN architecture"""
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values