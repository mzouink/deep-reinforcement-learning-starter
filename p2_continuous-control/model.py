import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1 = 96, fc2 = 96, seed=42):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.bn2 = nn.BatchNorm1d(fc2)
        self.fc3 = nn.Linear(fc2, action_size)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=0.01, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=0.01, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)


    def forward(self, state):
        x = self.bn0(state)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1 = 96, fc2 = 96, seed=42):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1+action_size, fc2)
        self.fc3 = nn.Linear(fc2, 1)
        self.initialize_weights()



    def initialize_weights(self):
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=0.01, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=0.01, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)


    def forward(self, state, action):
        state = self.bn0(state)
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        