# -*- coding: utf-8 -*-
# created by makise, 2022/2/24

# using pytorch to train a small feedforward neural network on subset of gtsrb dataset.
import numpy as np
import torch
import torch.nn as nn

# define the output size of the network
OUTPUT_SIZE = 7


class OcclusionLayer(nn.Module):
    def __init__(self, image, first_layer, dataset=None, shrink=True):
        super(OcclusionLayer, self).__init__()
        image_channel, image_height, image_width = image.shape
        self.fc1 = OcclusionFirstLayer(size_in=4, size_out=image_height * 2 + image_width * 2)
        self.fc2 = OcclusionSecondLayer(size_in=self.fc1.size_out, size_out=self.fc1.size_out // 2)
        self.fc3 = OcclusionThirdLayer(size_in=self.fc2.size_out, size_out=image_width * image_height * 2, image_shape=image.shape)
        self.fc4 = OcclusionFourthLayer(size_in=self.fc3.size_out, size_out=image_channel * image_width * image_height, image=image, model_first_layer=first_layer, shrink=shrink, dataset=dataset)


    def forward(self, x, epsilons):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x, epsilons))
        x = torch.relu(self.fc4(x))
        return x

class OcclusionFirstLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        weights, bias = self.init_weights_bias(size_in, size_out)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weights, x) + self.bias

    def init_weights_bias(self, size_in, size_out):
        weights = torch.zeros(size_out, size_in)
        bias = torch.zeros(size_out)

        # set the weight
        block_size = size_out // 4
        for i in range(4):
            if i == 0 or i == 2:
                for j in range(block_size):
                    weights[i * block_size + j, i] = 1
                    bias[i * block_size + j] = -(j + 1)
            elif i == 1 or i == 3:
                for j in range(block_size):
                    weights[i * block_size + j, i - 1] = -1
                    weights[i * block_size + j, i] = -1
                    bias[i * block_size + j] = j + 2

        return weights, bias


class OcclusionSecondLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        weights, bias = self.init_weights_bias(size_in, size_out)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weights, x) + self.bias

    def init_weights_bias(self, size_in, size_out):
        weights = torch.zeros(size_out, size_in)
        block_size = size_out // 2
        for i in range(2):
            for j in range(block_size):
                weights[i * block_size + j, 2 * i * block_size + j] = -1
                weights[i * block_size + j, 2 * i * block_size + block_size + j] = -1
        bias = torch.ones(size_out)

        return weights, bias


class OcclusionThirdLayer(nn.Module):
    def __init__(self, size_in, size_out, image_shape):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        _, image_height, image_width = image_shape
        weights, bias = self.init_weights_bias(size_in, size_out, image_shape)
        weights_eps, bias_eps = self.init_weight_bias_for_epsilons(size_in_eps=image_height * image_width, size_out=size_out, image_shape=image_shape)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.weights_eps = nn.Parameter(weights_eps, requires_grad=False)
        self.bias_eps = nn.Parameter(bias_eps, requires_grad=False)

    def forward(self, x, epsilon):
        return torch.matmul(self.weights, x) + self.bias + torch.matmul(self.weights_eps, epsilon) + self.bias_eps

    def init_weights_bias(self, size_in, size_out, image_shape):
        weights = torch.zeros(size_out, size_in)
        _, image_height, image_width = image_shape
        input_block_size = size_in // 2
        # output has only 1 part for occlusion
        for i in range(size_out):
            r, c = (i // 2) // image_width, (i // 2) % image_width
            weights[i, r] = 1
            weights[i, input_block_size + c] = 1

        bias = -torch.ones(size_out) * 2

        return weights, bias


    def init_weight_bias_for_epsilons(self, size_in_eps, size_out, image_shape):
        weights = torch.zeros(size_out, size_in_eps)
        _, image_height, image_width = image_shape
        for i in range(size_out):
            if i % 2 == 0:
                weights[i, (i // 2)] = 1
            else:
                weights[i, (i // 2)] = -1
        return weights, torch.zeros(size_out)


class OcclusionFourthLayer(nn.Module):
    def __init__(self, size_in, size_out, image, model_first_layer, shrink, dataset):
        super().__init__()
        if dataset not in ['gtsrb', 'mnist'] and dataset is not None:
            raise ValueError('dataset must be gtsrb or mnist')
        self.dataset = dataset
        self.size_in = size_in
        if shrink:
            self.size_out = model_first_layer.out_features
        else:
            self.size_out = image.shape[0] * image.shape[1] * image.shape[2]
        self.image = image
        weights, bias = self.init_weights_bias(size_in, size_out, image)
        if shrink:
            weights = torch.matmul(model_first_layer.weight, weights)
            bias = model_first_layer.bias + torch.matmul(model_first_layer.weight, bias)
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.matmul(self.weights, x) + self.bias

    def init_weights_bias(self, size_in, size_out, image):
        # assert image is a tensor
        assert isinstance(image, torch.Tensor)
        # flatten image into 1d
        image_flatten = image.view(-1)
        image_channel, image_height, image_width = image.shape
        weights = torch.zeros(size_out, size_in)
        if self.dataset == 'mnist':
            factor = 1.0 / 0.3081
        elif self.dataset == 'gtsrb':
            factor = 0.5 / 0.265
        else:
            factor = 1
        for channel in range(image_channel):
            for i in range(size_out // image_channel):
                weights[channel * image_height * image_width + i, i * 2] = factor
                weights[channel * image_height * image_width + i, i * 2 + 1] = -factor
        bias = torch.ones(size_out) * image_flatten

        return weights, bias
