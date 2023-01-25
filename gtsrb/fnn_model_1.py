# -*- coding: utf-8 -*-
# created by makise, 2022/2/24

# using pytorch to train a small feedforward neural network on subset of gtsrb dataset.


import torch
import torch.nn as nn

# define the output size of the network
OUTPUT_SIZE = 7

# this model has the same structure as the one in the DeepCert paper
class SmallDNNModel(nn.Module):
    def __init__(self):
        super(SmallDNNModel, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

