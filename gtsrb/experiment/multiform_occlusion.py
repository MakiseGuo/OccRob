# -*- coding: utf-8 -*-
# created by makise, 2022/8/26

import json

import sys

import copy
from datetime import datetime

import numpy as np
import os

import pebble
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
from maraboupy import Marabou, MarabouCore

import time
from gtsrb.gtsrb_dataset import GTSRB
from gtsrb.fnn_model_1 import SmallDNNModel as SmallModelGTSRB
from gtsrb.fnn_model_2 import SmallDNNModel2 as MediumModelGTSRB
from gtsrb.fnn_model_3 import SmallDNNModel3 as LargeModelGTSRB
from occlusion_layer.occlusion_layer_v4 import OcclusionLayer
from task import determine_robustness_with_epsilon
from utils.arg_util import ArgumentParser
from utils.log_util import Tee


class ExtendedModel(nn.Module):
    def __init__(self, occlusion_layer, origin_model):
        super(ExtendedModel, self).__init__()
        self.occlusion_layer = occlusion_layer
        self.extended_model = nn.Sequential(
            *list(origin_model.children())[1:]
        )

    def forward(self, x, epsilons):
        x = self.occlusion_layer(x, epsilons)
        x = self.extended_model(x)
        return x


def save_extended_model_onnx(image, model, idx):
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0], dataset='gtsrb')
    extended_model = ExtendedModel(occlusion_layer, model)
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = (torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.ones(32 * 32) * 0.01)
    tmp_dir = 'tmp/v4/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    onnx_model_filename = tmp_dir + 'extended_model_shrink_idx_{}.onnx'.format(idx)
    torch.onnx.export(extended_model, dummy_input, onnx_model_filename)
    return onnx_model_filename


def verify_with_marabou(model_filepath, label, a, b, size_a, size_b, epsilon):
    task_start_time = time.monotonic()
    network = Marabou.read_onnx(model_filepath)
    inputs = network.inputVars[0]
    epsilons = network.inputVars[1]
    outputs = network.outputVars
    n_outputs = outputs.flatten().shape[0]
    a_lower, a_upper = a
    b_lower, b_upper = b
    size_a_lower, size_a_upper = size_a
    size_b_lower, size_b_upper = size_b
    network.setLowerBound(inputs[0], a_lower)
    network.setUpperBound(inputs[0], a_upper)
    network.setLowerBound(inputs[1], size_a_lower)
    network.setUpperBound(inputs[1], size_a_upper)
    network.setLowerBound(inputs[2], b_lower)
    network.setUpperBound(inputs[2], b_upper)
    network.setLowerBound(inputs[3], size_b_lower)
    network.setUpperBound(inputs[3], size_b_upper)
    for i in range(len(epsilons)):
        tmp_a, tmp_b = i // 32, i % 32
        if tmp_a > a_lower and tmp_a < a_upper + size_a_upper and tmp_b > b_lower and tmp_b < b_upper + size_b_upper:
            network.setLowerBound(epsilons[i], -epsilon)
            network.setUpperBound(epsilons[i], epsilon)
        else:
            network.setLowerBound(epsilons[i], 0.0)
            network.setUpperBound(epsilons[i], 0.0)

    # for epsilon in epsilons:
    #     network.setLowerBound(epsilon, -0.5)
    #     network.setUpperBound(epsilon, 0.5)

    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs[i], outputs[label]], [1, -1], -1e-6)
    # output_constraints = []
    # for i in range(7):
    #     if i == label:
    #         continue
    #     eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    #     eq.addAddend(1, outputs[i])
    #     eq.addAddend(-1, outputs[label])
    #     eq.setScalar(0)
    #     output_constraints.append([eq])
    # network.addDisjunctionConstraint(output_constraints)

    options = Marabou.createOptions(solveWithMILP=True, verbosity=0)
    vals = network.solve(options=options)
    return vals[0], vals[1]


# Log both to console and file
def log_result(instrument, idx, dir):
    tmp = sys.stdout
    sys.stdout = Tee(sys.stdout, open(dir + 'experiment_instrument', 'a'))

    print("=" * 10 + f"Image {idx}: " + "=" * 10)

    """
    Show the instrument for Image idx as the following format:
    save_model_time(time for constructing the extended model): instrument['save_model_duration']
    verification_time(time for the whole verification process): instrument['verify_duration']
    is_robust(is the image robust): instrument['robust']
    timeout_proportion(timeout proportion of the whole verification process): instrument['timeout_proportion']
    """

    print("save_model_time: ", instrument['save_model_duration'])
    print("verification_time: ", instrument['verify_duration'])
    print("is_robust: ", instrument['robust'])
    print("timeout_proportion: ", instrument['timeout_proportion'])

    print("=" * 20)

    sys.stdout = tmp


def log_summary_result(results):
    for key, value in results.items():
        print("====== summary data for epsilon = {} ======".format(key))
        unrobust_count = 0
        robust_count = 0
        unrobust_time = 0
        robust_time = 0
        build_model_time = 0
        timeout_proportion = 0
        for entry in value:
            if entry['robust'] == False:
                unrobust_count += 1
                unrobust_time += entry['verify_duration']
            else:
                robust_count += 1
                robust_time += entry['verify_duration']
            build_model_time += entry['save_model_duration']
            timeout_proportion += entry['timeout_proportion']
        print("unrobust count: ", unrobust_count)
        print("robust count: ", robust_count)
        if unrobust_count == 0:
            print("unrobust average time: null")
        else:
            print("unrobust avergae time: ", unrobust_time / unrobust_count)
        if robust_count == 0:
            print("robust average time: null")
        else:
            print("robust average time: ", robust_time / robust_count)
        print("build model average time: ", build_model_time / len(value))
        print("timeout proportion: ", timeout_proportion / len(value))
        print("total count: ", len(value), flush=True)


def main(args):
    # print args
    print("=" * 20)
    print('Arguments:', flush=True)
    for arg in vars(args):
        print(arg, getattr(args, arg), flush=True)
    print("=" * 20)

    current_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    result_dir = 'experiment_result/multiform_occlusion_experiment_model_{}_size_{}_epsilon_{}_{}_{}/'.format(
        args.model,
        args.size,
        args.epsilon,
        'sorted' if args.sort == 1 else 'unsorted',
        current_timestamp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result = []
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
        is_robust = True

        total_time_start = time.monotonic()
        image, label = iter_on_loader.next()

        image = image.reshape(3, 32, 32)
        label = label.item()

        labels = model(image)
        labels = labels[0].argsort(descending=True)
        spurious_labels = []
        if labels[0] != label:
            print("image {} classified wrong".format(i), flush=True)
            continue
        for l in labels:
            if l.item() != label:
                spurious_labels.append(l.item())

        if args.sort == 0:
            spurious_labels = []
            for j in range(7):
                if j != label:
                    spurious_labels.append(j)

        save_model_start = time.monotonic()
        model_filepath = save_extended_model_onnx(image, model, i)
        save_model_duration = time.monotonic() - save_model_start
        instrument['save_model_duration'] = save_model_duration

        robust, adversarial_example, timeout_prop, verify_duration = determine_robustness_with_epsilon((args.size, args.size),
                                                                                                       spurious_labels,
                                                                                                       args.epsilon,
                                                                                                       model_filepath,
                                                                                                       verify_with_marabou,
                                                                                                       args.workers,
                                                                                                       args.split)
        instrument['verify_duration'] = verify_duration
        instrument['robust'] = robust
        instrument['adversarial_example'] = adversarial_example
        instrument['timeout_proportion'] = timeout_prop
        log_result(instrument, i, result_dir)
        result.append(instrument)

    with open(result_dir + 'result.json', 'w') as f:
        json.dump(result, f)
        f.write('\n')
        f.flush()

    return {
        'epsilon': args.epsilon,
        'result': result
    }


if __name__ == '__main__':

    """
    Optional Command Line Arguments:
    --model: small / medium / large
    --testNum: number of test images
    --epsilon: 0.05 / 0.1 / 0.2 / 0.3 / 0.4
    --size: size of occlusion
    --sort: whether using label sorting, default is True
    """
    parser = ArgumentParser(dataset='gtsrb', type='multiform')
    args = parser.parse_args()

    results = {}
    with pebble.ProcessPool(1) as pool:
        future = pool.map(main, [args])
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
                results[result['epsilon']] = result['result']
            except StopIteration:
                break
            except Exception:
                break

    log_summary_result(results)
