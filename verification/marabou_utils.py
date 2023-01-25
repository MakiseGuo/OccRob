# -*- coding: utf-8 -*-
# created by makise, 2022/2/27

# Some useful functions for Marabou

from maraboupy import Marabou, MarabouNetwork
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data

from gtsrb.gtsrb_dataset import GTSRB
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# load a network from onnx format file and return MarabouNetwork
def load_network(filename) -> MarabouNetwork:
    """
    Load a network from onnx format file and return MarabouNetwork
    :param filename: the name of the onnx file
    :return: network: MarabouNetwork
    """
    # the path of network file
    path = '../model/' + filename
    # load the network
    network = Marabou.read_onnx(path)
    return network


def load_sample_image() -> np.ndarray:
    """
    Load one sample image and resize it to 32*32.
    :return: numpy array of image
    """
    # load one image from training set using PIL and convert it to np array
    image = Image.open("../data/GTSRB/trainingset/00000/00000_00000.ppm")
    # resize the image to (32, 32), if it's less than 32, pad it with black pixels
    image = image.resize((32, 32), Image.ANTIALIAS)
    np_img = np.array(image)
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np.reshape(np_img, (1, 3, 32, 32))
    return np_img


def get_test_images_loader(input_size, batch_size=1, output_dim=43, classes=range(0, 43)) -> DataLoader:
    """
    Use Pytorch to load test images
    :param input_size: the size of input image, 2-tuple
    :return: data_loader: Pytorch DataLoader
    """
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])
    test_data = GTSRB(root_dir='../data', train=False, transform=data_transform, classes=classes)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    samples, labels = iter(test_loader).next()
    print("samples shape: ", samples.shape)

    return test_loader


def get_mnist_test_images_loader(batch_size=1, output_dim=10) -> DataLoader:
    """
    Use Pytorch to load test images
    :param input_size: the size of input image, 2-tuple
    :return: data_loader: Pytorch DataLoader
    """
    test_loader = data.DataLoader(
        datasets.MNIST('../mnist/data', train=False, transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)

    return test_loader