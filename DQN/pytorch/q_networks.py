import torch.nn as nn


class DQNEstimator(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )
    
    def forward(self, x):
        return self.model(x)