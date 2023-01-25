# -*- coding: utf-8 -*-
# created by makise, 2022/7/28
import json

import sys
from datetime import datetime

import copy
import numpy as np
import os

import pebble
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
from matplotlib import pyplot as plt
from maraboupy import Marabou, MarabouCore

import time

from utils.arg_util import ArgumentParser
from utils.log_util import Tee
from mnist.fnn_model_1 import FNNModel1 as SmallModelMnist
from mnist.fnn_model_2 import FNNModel1 as MediumModelMnist
from mnist.fnn_model_3 import FNNModel1 as LargeModelMnist
from occlusion_layer.occlusion_layer_v4 import OcclusionLayer
from task import determine_robustness_with_epsilon


class ExtendedSmallModel(nn.Module):
    def __init__(self, occlusion_layer, origin_model):
        super(ExtendedSmallModel, self).__init__()
        self.occlusion_layer = occlusion_layer
        self.extended_model = nn.Sequential(
            *list(origin_model.children())[1:]
        )

    def forward(self, x, epsilons):
        x = self.occlusion_layer(x, epsilons)
        x = self.extended_model(x)
        return x
    # def __init__(self, occlusion_layer, origin_model):
    #     super(ExtendedSmallModel, self).__init__()
    #     self.occlusion_layer = occlusion_layer
    #     self.origin_model = origin_model
    #
    # def forward(self, x, epsilons):
    #     x = self.occlusion_layer(x, epsilons)
    #     x = torch.relu(self.origin_model.fc2(x))
    #     x = self.origin_model.fc3(x)
    #     return x


class ExtendedMediumModel(nn.Module):
    def __init__(self, occlusion_layer, origin_model):
        super(ExtendedMediumModel, self).__init__()
        self.occlusion_layer = occlusion_layer
        self.extended_model = nn.Sequential(
            *list(origin_model.children())[1:]
        )

    def forward(self, x, epsilons):
        x = self.occlusion_layer(x, epsilons)
        x = self.extended_model(x)
        return x


class ExtendedLargeModel(nn.Module):
    def __init__(self, occlusion_layer, origin_model):
        super(ExtendedLargeModel, self).__init__()
        self.occlusion_layer = occlusion_layer
        self.extended_model = nn.Sequential(
            *list(origin_model.children())[1:]
        )

    def forward(self, x, epsilons):
        x = self.occlusion_layer(x, epsilons)
        x = self.extended_model(x)
        return x
    # def __init__(self, occlusion_layer, origin_model):
    #     super(ExtendedLargeModel, self).__init__()
    #     self.occlusion_layer = occlusion_layer
    #     self.origin_model = origin_model
    #
    # def forward(self, x, epsilons):
    #     x = self.occlusion_layer(x, epsilons)
    #     x = torch.relu(self.origin_model.fc2(x))
    #     x = torch.relu(self.origin_model.fc3(x))
    #     x = torch.relu(self.origin_model.fc4(x))
    #     x = torch.relu(self.origin_model.fc5(x))
    #     x = self.origin_model.fc6(x)
    #     return x


def show_occluded_image():
    model = SmallModelMnist()
    model.load_state_dict(torch.load('../../model/fnn_model_mnist_1.pth', map_location=torch.device('cpu')))
    image, label = get_a_test_image()
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0], dataset='mnist')
    input = torch.tensor([10.0, 10.0, 10.0, 10.0])
    epsilons = torch.ones(28 * 28) * -0.1
    occluded_image = occlusion_layer(input, epsilons)
    plt.subplot(1, 2, 1)
    # convert torch into numpy array
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(28, 28)
    plt.imshow(occluded_image)
    plt.subplot(1, 2, 2)
    image = image.numpy()
    image = image.reshape(28, 28)
    plt.imshow(image)
    plt.show()


def save_extended_model_onnx(image, model, model_size):
    occlusion_layer = OcclusionLayer(image=image, first_layer=list(model.children())[0], dataset='mnist')
    if model_size == 'small':
        extended_model = ExtendedSmallModel(occlusion_layer, model)
    elif model_size == 'medium':
        extended_model = ExtendedMediumModel(occlusion_layer, model)
    elif model_size == 'large':
        extended_model = ExtendedLargeModel(occlusion_layer, model)
    else:
        raise ValueError('model size should be small, medium or large')
    extended_model = extended_model.to(torch.device('cpu'))
    dummy_input = (torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.ones(28 * 28) * 0.01)
    onnx_model_filename = 'tmp/v4/' + 'fnn_model_mnist_3_extended_shrink.onnx'
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
        tmp_a, tmp_b = i // 28, i % 28
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


def get_a_test_image(idx):
    test_loader = data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_test_loader = iter(test_loader)
    # iter_test_loader.next()
    # iter_test_loader.next()
    image, label = iter_test_loader.next()
    for i in range(idx):
        image, label = iter_test_loader.next()
    image = image.reshape(1, 28, 28)
    return image.numpy(), label.item()

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
        is_robust = True

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

        if args.sort == 0:
            spurious_labels = []
            for j in range(10):
                if j != label:
                    spurious_labels.append(j)

        save_model_start = time.monotonic()
        model_filepath = save_extended_model_onnx(image, model, args.model)
        save_model_duration = time.monotonic() - save_model_start
        instrument['save_model_duration'] = save_model_duration

        robust, adversarial_example, timeout_prop, verify_duration = determine_robustness_with_epsilon((args.size, args.size),
                                                                                      spurious_labels, args.epsilon,
                                                                                      model_filepath,
                                                                                      verify_with_marabou,
                                                                                      args.workers, args.split)
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
    parser = ArgumentParser(dataset='mnist', type='multiform')
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