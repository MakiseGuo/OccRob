# -*- coding: utf-8 -*-
# created by makise, 2022/2/24

# using pytorch to train a small feedforward neural network on subset of gtsrb dataset.


import torch
import torch.nn as nn

# define the output size of the network
OUTPUT_SIZE = 7

# this model has the same structure as the one in the DeepCert paper
class SmallDNNModel2(nn.Module):
    def __init__(self):
        super(SmallDNNModel2, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 43)
        self.fc4 = nn.Linear(43, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

