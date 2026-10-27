"""Original MIMO architecture from Adaptive Changing Server.py."""
import torch
from torch import nn

class MIMOModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, output_size=10):
        super(MIMOModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

