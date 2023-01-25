# -*- coding: utf-8 -*-
# created by makise, 2022/8/4

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from maraboupy import Marabou, MarabouCore
from torchvision import transforms

from gtsrb.fnn_model_1 import SmallDNNModel as SmallModelGTSRB
from gtsrb.fnn_model_2 import SmallDNNModel2 as MediumModelGTSRB
from gtsrb.fnn_model_3 import SmallDNNModel3 as LargeModelGTSRB
from gtsrb.gtsrb_dataset import GTSRB
from occlusion_layer.occlusion_layer_v3 import OcclusionLayer
from task import determine_robustness_color_fixed
from utils.arg_util import ArgumentParser
from utils.log_util import Tee

result = []

def save_extended_model_onnx(image, model, idx):
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0])
    extended_model = nn.Sequential(
        occlusion_layer,
        *list(model.children())[1:]
    )
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    tmp_dir = 'tmp/v3/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    onnx_model_filename = tmp_dir + 'extended_model_shrink_idx_{}.onnx'.format(idx)
    torch.onnx.export(extended_model, dummy_input, onnx_model_filename)
    return onnx_model_filename

def verify_with_marabou(model_filepath, label, a, b, size_a, size_b, color):
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
    network.setLowerBound(inputs[4], color_lower[0])
    network.setUpperBound(inputs[4], color_upper[0])
    network.setLowerBound(inputs[5], color_lower[1])
    network.setUpperBound(inputs[5], color_upper[1])
    network.setLowerBound(inputs[6], color_lower[2])
    network.setUpperBound(inputs[6], color_upper[2])

    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs[i], outputs[label]], [1, -1], -1e-6)

    options = Marabou.createOptions(solveWithMILP=True, verbosity=0)
    vals = network.solve(options=options)

    return vals[0], vals[1]


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
    parser = ArgumentParser(dataset='gtsrb', type='uniform')
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
        model = SmallModelGTSRB()
        model_name = 'fnn_model_gtsrb_small_1_different_class.pth'
    elif args.model == 'medium':
        model = MediumModelGTSRB()
        model_name = 'fnn_model_gtsrb_small_2.pth'
    elif args.model == 'large':
        model = LargeModelGTSRB()
        model_name = 'fnn_model_gtsrb_small_3.pth'
    else:
        raise ValueError('Invalid model name')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])
    test_data = GTSRB(root_dir='../../data/', train=False, transform=transform, classes=[1, 2, 3, 4, 5, 7, 8])
    test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    iter_on_loader = iter(test_loader)
    model.load_state_dict(torch.load('../../model/' + model_name, map_location=torch.device('cpu')))
    for i in range(args.testNum):
        print("=" * 20)
        print("image {}:".format(i))
        instrument = {}

        total_time_start = time.monotonic()
        image, label = iter_on_loader.next()

        image = image.reshape(3, 32, 32)
        label = label.item()

        labels = model(image)
        labels = labels[0].argsort(descending=True)
        if labels[0] != label:
            print("image {} classified wrong".format(i), flush=True)
            continue
        spurious_labels = []
        for l in labels:
            if l.item() != label:
                spurious_labels.append(l.item())

        save_model_start = time.monotonic()
        model_filepath = save_extended_model_onnx(image, model, i)
        save_model_duration = time.monotonic() - save_model_start
        instrument['save_model_duration'] = save_model_duration

        verify_start = time.monotonic()
        robusts, adversarial_examples, size_times, timeout_prop = determine_robustness_color_fixed((1, 10), spurious_labels, model_filepath, args.color, verify_with_marabou)
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