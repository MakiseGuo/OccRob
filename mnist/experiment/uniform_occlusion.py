# -*- coding: utf-8 -*-
# created by makise, 2022/7/14
import json
import sys
import time

import numpy as np
from datetime import datetime

import os
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from maraboupy import Marabou, MarabouCore

from utils.arg_util import ArgumentParser
from utils.log_util import Tee
from mnist.fnn_model_1 import FNNModel1 as SmallModelMnist
from mnist.fnn_model_2 import FNNModel1 as MediumModelMnist
from mnist.fnn_model_3 import FNNModel1 as LargeModelMnist
from occlusion_layer.occlusion_layer_v3 import OcclusionLayer
from task import determine_robustness_color_fixed
import shutil

result = []

def save_extended_model_onnx(image, model, idx):
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0])
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())[1:]
    )
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    tmp_dir = 'tmp/v3/'
    # create tmp dir if not exist
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    onnx_model_filename = tmp_dir + 'extended_model_shrink_idx_{}.onnx'.format(idx)
    torch.onnx.export(extended_model, dummy_input, onnx_model_filename)
    return onnx_model_filename


def cp_model_file(model_filepath):
    model_filepaths = [model_filepath]
    for i in range(1, 4):
        path = 'tmp/v3/' + model_filepath.split('.')[0] + '_{}.onnx'.format(i)
        shutil.copy(model_filepath, path)
        model_filepaths.append(path)
    return model_filepaths


def verify_with_marabou(model_filepath, label, a, b, size_a, size_b, color):
    # signal.signal(signal.SIGALRM, handle_timeout)
    # print("start verification {} {} {} {} {} {}".format(a, b, size_a, size_b, color, label), flush=True)
    network = Marabou.read_onnx(model_filepath)
    inputs = network.inputVars[0]
    outputs = network.outputVars
    n_outputs = outputs.flatten().shape[0]
    a_lower, a_upper = a
    b_lower, b_upper = b
    size_a_lower, size_a_upper = size_a
    size_b_lower, size_b_upper = size_b
    color_lower, color_upper = color
    network.setLowerBound(inputs[0], a_lower)
    network.setUpperBound(inputs[0], a_upper)
    network.setLowerBound(inputs[1], size_a_lower)
    network.setUpperBound(inputs[1], size_a_upper)
    network.setLowerBound(inputs[2], b_lower)
    network.setUpperBound(inputs[2], b_upper)
    network.setLowerBound(inputs[3], size_b_lower)
    network.setUpperBound(inputs[3], size_b_upper)
    network.setLowerBound(inputs[4], color_lower)
    network.setUpperBound(inputs[4], color_upper)

    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs[i], outputs[label]], [1, -1], -1e-6)

    options = Marabou.createOptions(solveWithMILP=False, verbosity=0)
    vals = network.solve(verbose=True, options=options)

    return vals[0], vals[1]


def handle_timeout():
    raise TimeoutError('verify over 1 min')


"""
Log the instrument both to stdout and to a file
"""
def log_result(instrument, idx, dir):
    tmp = sys.stdout
    sys.stdout = Tee(sys.stdout, open(dir + 'experiment_instrument', 'a'))

    print("=" * 10 + f"Image {idx}: " + "=" * 10)

    """
    Show the instrument for Image idx as the following format:
    save_model_time(time for constructing the extended model): instrument['save_model_duration']
    verification_time(time for the whole verification process): instrument['verify_duration']
    A list for robustness and verification_time for 1x1 to 10x10 size:
    - size: 1x1 is_robust: instrument['robusts'][0] instrument['size_times'][0]
    ...
    - size: 10x10 is_robust: instrument['robusts'][9] instrument['size_times'][9]
    timeout_proportion(timeout proportion of the whole verification process): instrument['timeout_proportion']
    """

    print("save_model_time: ", instrument['save_model_duration'])
    print("verification_time: ", instrument['verify_duration'])
    for i in range(len(instrument['robusts'])):
        print(f"size: {i + 1}x{i + 1} is_robust: {instrument['robusts'][i]} {instrument['size_times'][i]}")
    print("timeout_proportion: ", instrument['timeout_proportion'])

    print("=" * 20)

    sys.stdout = tmp


if __name__ == '__main__':

    """
    Optional Command Line Arguments:
    --model: small / medium / large
    --color: 0-1
    --testNum: number of test images
    """
    parser = ArgumentParser(dataset='mnist', type='uniform')
    args = parser.parse_args()
    # print args with detailed information
    parser.log_args()

    current_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    result_dir = 'experiment_result/uniform_occlusion_model_{}_{}/'.format(args.model, current_timestamp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    model = None
    model_name = ''
    if args.model == 'small':
        model = SmallModelMnist()
        model_name = 'fnn_model_mnist_1.pth'
    elif args.model == 'medium':
        model = MediumModelMnist()
        model_name = 'fnn_model_mnist_2.pth'
    elif args.model == 'large':
        model = LargeModelMnist()
        model_name = 'fnn_model_mnist_3.pth'
    else:
        raise ValueError('Invalid model name')

    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_on_loader = iter(test_loader)
    model.load_state_dict(torch.load('../../model/' + model_name, map_location=torch.device('cpu')))
    for i in range(args.testNum):
        print("=" * 20)
        print("image {}:".format(i))
        instrument = {}

        total_time_start = time.monotonic()
        image, label = iter_on_loader.next()

        image = image.reshape(1, 28, 28)
        label = label.item()

        labels = model(image)
        labels = labels[0].argsort(descending=True)
        spurious_labels = []
        for l in labels:
            if l.item() != label:
                spurious_labels.append(l.item())

        # print("spurious label: ", spurious_labels)
        save_model_start = time.monotonic()
        model_filepath = save_extended_model_onnx(image, model, i)
        # model_filepaths = cp_model_file(model_filepath)
        save_model_duration = time.monotonic() - save_model_start
        instrument['save_model_duration'] = save_model_duration

        verify_start = time.monotonic()
        robusts, adversarial_examples, size_times, timeout_prop = determine_robustness_color_fixed((1, 10), spurious_labels, model_filepath, args.color,
                                                                             verify_with_marabou)
        verify_duration = time.monotonic() - verify_start
        instrument['verify_duration'] = verify_duration
        instrument['robusts'] = robusts
        instrument['size_times'] = size_times
        instrument['timeout_proportion'] = timeout_prop
        instrument['adversarial_examples'] = adversarial_examples
        result.append(instrument)

        log_result(instrument, i, result_dir)

    with open(result_dir + 'result.json', 'w') as f:
        json.dump(result, f)
        f.write('\n')
        f.flush()
