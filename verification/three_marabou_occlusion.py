# -*- coding: utf-8 -*-
# created by makise, 2022/3/8

# corresponding to the thought #3 in the doc
# given occlusion color, given occlusion size, verify that no matter where the occlusion area is
# network can classify correctly
import argparse
import json
import os

import onnx
import onnxruntime
from maraboupy import Marabou, MarabouNetwork, MarabouCore
from PIL import Image
import numpy as np
import torch
import time

import pebble

from marabou_utils import load_network, load_sample_image, get_test_images_loader, get_mnist_test_images_loader
from occlusion_bound import calculate_entire_bounds
from interpolation import occlusion

# define some global variables
occlusion_size = (2, 2)
occlusion_color = 0
batch_num = 10
result_file_dir = '/home/GuoXingWu/pycharm_project_368/experiment/results/thought_3/'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
use_marabou = True

epsilon = 1e-3


def verify_occlusion_by_dividing(image: np.array, label: int, occlusion_size: tuple, occlusion_color: int, block_size):
    """
    given an image, label, occlusion size and occlusion color, verify that no matter where the occlusion area is
    network can classify correctly
    :param image: image in np array after normalization, size: 1*3*32*32
    :param label: int indicates the correct label
    :param occlusion_size: tuple indicates the occlusion size
    :param occlusion_color: int indicates the occlusion color
    :param block_size: int indicates the block size
    :return: vals, constraints_calculation_time, verify_time
    """
    image = image[0]
    c, h, w = image.shape
    # divide the image into several parts, solve them separately
    block_num = (h // block_size[0], w // block_size[1])

    total_constraints_calculation_time = 0
    total_verify_time = 0
    vals = ['unsat']

    with pebble.ProcessPool(1) as pool:
        for i in range(block_num[0]):
            print("current block: ", i, i)
            for l in range(output_dim):
                print("current label: ", l)
                if l == label:
                    continue
                parameter = (image, l, occlusion_size, occlusion_color, (block_size[0] + occlusion_size[0], block_size[1] + occlusion_size[1]), i * block_size[0], i * block_size[1])
                future = pool.schedule(verify_occlusion_with_fixed_size, parameter, timeout=60)
                verify_time = 60
                constraints_calculation_time = 0
                try:
                    sat, result, constraints_calculation_time, verify_time = future.result()
                except Exception as error:
                    print("timeout or error occurred for block: ", i, i)
                    print(error)
                    sat = 'unsat'
                total_constraints_calculation_time += constraints_calculation_time
                total_verify_time += verify_time
                if sat == 'sat':
                    return [sat, result], total_constraints_calculation_time, total_verify_time

    return vals, total_constraints_calculation_time, total_verify_time


# thought #3 in the doc
def verify_occlusion_with_fixed_size(image: np.array, label: int, occlusion_size: tuple, occlusion_color: int,
                                     block_size: tuple, width_offset, height_offset):
    """
    given an image, label, occlusion size and occlusion color, verify that no matter where the occlusion area is
    network can classify correctly
    :param image: image in np array after normalization, size: 3*32*32
    :param label: int indicates the correct label
    :param occlusion_size: tuple indicates the occlusion size
    :param occlusion_color: int indicates the occlusion color
    :param block_size: tuple indicates the size of each block
    :param width_offset: int indicates the offset of the occlusion area on the width axis
    :param height_offset: int indicates the offset of the occlusion area on the height axis
    :return: vals, constraints_calculation_time, verify_time
    """
    constraints_calculation_start_time = time.monotonic()
    # load network
    network = load_network(model_name)

    inputs = network.inputVars[0][0]  # 3*32*32 or 28*28
    # only fot mnist
    if inputs.shape[0] == 28:
        inputs = np.reshape(inputs, (1, 28, 28))
    outputs = network.outputVars[0]  # {output_dim}
    n_outputs = outputs.flatten().shape[0]
    assert image.shape == inputs.shape
    assert n_outputs == output_dim

    c, h, w = image.shape
    occlusion_height, occlusion_width = occlusion_size

    # define the constraints on the entire image
    # constraints = calculate_constrains(image, inputs)
    x = network.getNewVariable()
    y = network.getNewVariable()
    x_lower_bound = max(0, width_offset - occlusion_width + 1)
    x_upper_bound = min(block_size[1] - occlusion_width + width_offset, h)
    y_lower_bound = max(0, height_offset - occlusion_height + 1)
    y_upper_bound = min(block_size[0] - occlusion_height + height_offset, w)
    network.setLowerBound(x, x_lower_bound)
    network.setUpperBound(x, x_upper_bound)
    network.setLowerBound(y, y_lower_bound)
    network.setUpperBound(y, y_upper_bound)
    print(f'x: [{x_lower_bound}, {x_upper_bound}]', )

    # iterate over the target block of image
    for i in range(height_offset, min(height_offset + block_size[0], h)):
        for j in range(width_offset, min(width_offset + block_size[1], w)):
            # print(f'current pixel: {i}, {j}')
            # occlusion point cover (i, j)
            # the constraints should have size like [[eq1, eq2], [eq3, eq4], ...]
            # stand for (eq1 and eq2) or (eq3 and eq4) or ...
            # network.addDisjunctionConstraint(constraints)
            constraints = []
            # this equation is like (x <= j) and (x >= j - occlusion_size[0] - 1) and (y <= i) and
            # (y >= i - occlusion_size[1] - 1) and (image[i, j] == occlusion_color)
            # with the simple occlusion_size = (1, 1), inequality becomes equality
            eqs = []
            eq1 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq1.addAddend(1, x)
            eq1.setScalar(j + 1 - epsilon)
            eqs.append(eq1)
            eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq2.addAddend(1, x)
            eq2.setScalar(j - occlusion_width + 1)
            eqs.append(eq2)
            eq3 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq3.addAddend(1, y)
            eq3.setScalar(i + 1 - epsilon)
            eqs.append(eq3)
            eq4 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq4.addAddend(1, y)
            eq4.setScalar(i - occlusion_height + 1)
            eqs.append(eq4)
            for k in range(c):
                eq5 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq5.addAddend(1, inputs[k][i][j])
                eq5.setScalar(((occlusion_color / 255.0) - mean[k]) / std[k])
                eqs.append(eq5)
            constraints.append(eqs)
            # print(f'x <= {j + 1 - epsilon} and x >= {j - occlusion_width + 1} and y <= {i + 1 - epsilon} and '
            #       f'y >= {i - occlusion_height + 1} and image[{i}, {j}] == {occlusion_color}')
            # otherwise
            # since don't know how to write unequal constraints
            # change two unequal constraints into four greater equal and less equal constraints
            # and also don't know if there exists greater and less relation
            # this equation is like (x >= j) and image[i, j] == origin_color
            # this has four similar constraints connecting with or relation
            # the other three has the same second part and the first part is separately
            # (x <= j - occlusion_size[0] - 1) and (j >= i) and (y <= i - occlusion_size[1] - 1)
            eqs = []
            for k in range(c):
                eq10 = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq10.addAddend(1, inputs[k][i][j])
                eq10.setScalar(image[k][i][j])
                eqs.append(eq10)
            eqs_temp = eqs.copy()
            eq6 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq6.addAddend(1, x)
            eq6.setScalar(j + 1)
            eqs.append(eq6)
            constraints.append(eqs)
            eqs = eqs_temp.copy()
            eq7 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq7.addAddend(1, x)
            eq7.setScalar(j - occlusion_width + 1 - epsilon)
            eqs.append(eq7)
            constraints.append(eqs)
            eqs = eqs_temp.copy()
            eq8 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq8.addAddend(1, y)
            eq8.setScalar(i + 1)
            eqs.append(eq8)
            constraints.append(eqs)
            eqs = eqs_temp.copy()
            eq9 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq9.addAddend(1, y)
            eq9.setScalar(i - occlusion_height + 1 - epsilon)
            eqs.append(eq9)
            constraints.append(eqs)
            # print(f'(x >= {j + 1} or x <= {j - occlusion_width + 1 - epsilon} or y >= {i + 1} or'
            #       f' y <= {i - occlusion_height + 1 - epsilon}) and image[{i}, {j}] == origin_color')
            # add constraints to network
            network.addDisjunctionConstraint(constraints)

    lower_bound = (0 - mean) / std
    upper_bound = (1 - mean) / std
    # add additional bounds for the inputs
    fixed_pixels = 0
    for i in range(h):
        for j in range(w):
            for k in range(c):
                # set a fixed value for all pixel not in the target block
                if i < height_offset or i >= height_offset + block_size[0] or j < width_offset or j >= width_offset + \
                        block_size[1]:
                    network.setLowerBound(inputs[k][i][j], image[k][i][j])
                    network.setUpperBound(inputs[k][i][j], image[k][i][j])
                    fixed_pixels += 1
                else:
                    # must add a small value to avoid constraints conflict issues
                    network.setLowerBound(inputs[k, i, j], lower_bound[k] - 0.001)
                    network.setUpperBound(inputs[k, i, j], upper_bound[k] + 0.001)
    print("fixed pixels: ", fixed_pixels)

    # add bounds to output
    # new output constraints using disjunction constraints
    # output_constraints = []
    # for i in range(output_dim):
    #     if i == label:
    #         continue
    #     eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    #     eq.addAddend(1, outputs[i])
    #     eq.addAddend(-1, outputs[label])
    #     eq.setScalar(0)
    #     output_constraints.append([eq])
    # network.addDisjunctionConstraint(output_constraints)
    # origin output constraints
    for i in range(n_outputs):
        if i != label:
            network.addInequality([outputs[i], outputs[label]], [1, -1], 0)
    constraints_calculation_end_time = time.monotonic()
    constraints_calculation_time = constraints_calculation_end_time - constraints_calculation_start_time

    verify_start_time = time.monotonic()
    print("verify start: current true label: ", label, flush=True)
    options = Marabou.createOptions(solveWithMILP=True)
    vals = network.solve(verbose=True, options=options)
    verify_end_time = time.monotonic()
    verify_time = verify_end_time - verify_start_time

    # print("vals length: ", len(vals), flush=True)

    return vals[0], vals[1], constraints_calculation_time, verify_time


def traverse_occlusion_with_fixed_size_by_onnx(image, label, occlusion_size, occlusion_color):
    """
    Use onnxruntime to traversal all possible image with given occlusion size as a benchmark for verification
    :param image: 1*3*32*32 image in np array after normalization with resize and normalization
    :param label: int indicates the correct label
    :param occlusion_size: tuple indicates the occlusion size
    :param occlusion_color: int indicates the occlusion color
    :return: robust, traversal_time
    """
    start_time = time.monotonic()
    robust = True
    adv_num = 0
    sample_num = 0
    image = np.transpose(image[0], (1, 2, 0))
    h, w, c = image.shape
    print(f'height: {h}, width: {w}')
    # denormalize the image with given mean and std
    image = (image * std + np.array((0.3337, 0.3064, 0.3171))) * 255
    # load onnx model
    onnx_model_path = "../model/" + model_name
    onnx_model = onnx.load(onnx_model_path)
    # create the onnxruntime session
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # create the input tensor
    input_name = ort_session.get_inputs()[0].name
    # iterate on the whole image
    print("occlusion size: ", occlusion_size)
    for i in range(h - occlusion_size[0] + 1):
        for j in range(w - occlusion_size[1] + 1):
            occluded_image = occlusion.occlusion_with_interpolation(image, (i, j), occlusion_size, occlusion_color)
            occluded_image = np.clip(occluded_image, 0, 255).astype(np.uint8)
            # normalize image
            occluded_image = (occluded_image / 255.0 - mean) / std
            occluded_image = np.transpose(occluded_image, (2, 0, 1))
            occluded_image = np.reshape(occluded_image, (1, 3, 32, 32))
            input_tensor = occluded_image.astype(np.float32)
            # run the model
            output_tensor = ort_session.run(None,
                                            {input_name: input_tensor})  # the torch_out is 1 * batch_size * output_dim
            output_tensor = torch.tensor(output_tensor[0])
            _, predicted = torch.max(output_tensor, 1)
            sample_num += 1
            if predicted[0] != label:
                robust = False
                adv_num += 1
    return robust, adv_num, sample_num, time.monotonic() - start_time


def calculate_constrains(image, inputs):
    """
    calculate the constraints for the entire image
    :return: constraints
    """
    c, h, w = image.shape
    constraints = []
    test_constraints = []
    for i in range(h):
        for j in range(w):
            eqs = []
            test_eqs = []
            for k in range(c):
                eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
                eq.addAddend(1, inputs[k, i, j])
                eq.setScalar(occlusion_color)
                eqs.append(eq)
                test_eq = f'x({k}, {i}, {j}) = {occlusion_color}'
                test_eqs.append(test_eq)
            for ii in range(h):
                for jj in range(w):
                    if i == ii and j == jj:
                        continue
                    for kk in range(c):
                        eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
                        eq.addAddend(1, inputs[kk, ii, jj])
                        eq.setScalar(image[kk, ii, jj])
                        eqs.append(eq)
                        test_eq = f'x({kk}, {ii}, {jj}) = {image[kk, ii, jj]}'
                        test_eqs.append(test_eq)
            constraints.append(eqs)
            test_constraints.append(test_eqs)
    print("test_constraints length: ", len(test_constraints))
    print("test_constraints[0] length: ", len(test_constraints[0]))
    return constraints


def conduct_experiment(occlusion_size, occlusion_color, block_size, pe_timestamp):
    print("ThreeMarabouOcclusionExperiment: experiment start, occlusion size: ", occlusion_size,
          ", block size: ", block_size)
    img_loader = get_test_images_loader(input_size, output_dim=output_dim, classes=[1, 2, 3, 4, 5, 7, 8])
    iterable_img_loader = iter(img_loader)

    results = []
    for i in range(batch_num):
        start_time = time.monotonic()
        image, label = iterable_img_loader.next()
        image = image.numpy()
        label = label.item()
        isRobust = True
        # constraints_calculation_time = -1.0
        # verify_time = -1.0
        predicted_label = -1
        results_batch = []
        adversarial_example = None
        adv_example_list = None

        vals, constraints_calculation_time, verify_time = verify_occlusion_by_dividing(image, label,
                                                                                       occlusion_size,
                                                                                       occlusion_color,
                                                                                       block_size)
        results_batch.append(
            {'vals': vals[0], 'constraints_calculation_time': constraints_calculation_time,
             'verify_time': verify_time,
             })
        if vals[0] == 'sat':
            # adversarial_example = vals[1]
            # unpack adversarial example to a list
            # adversarial_example is a dict{int, float}
            # key is the index of the variable in the network
            # value is the value of the variable
            # adv_example_list = [adversarial_example[i] for i in range(channel * input_size[0] * input_size[1])]
            isRobust = False
        total_time = time.monotonic() - start_time

        results.append(
            {'robust': isRobust, 'total_verify_time': total_time,
             'true_label': label, 'predicted_label': predicted_label, 'adv_example': adv_example_list,
             'origin_image': image.tolist(), 'detail': results_batch})

        dir = result_file_dir + f'pe_{pe_timestamp}/'
        # create directory if not exist
        if not os.path.exists(dir):
            os.makedirs(dir)
        # save results to file
        result_filepath = dir + f'{model_name}_batchNum_{batch_num}_occlusionSize_{occlusion_size[0]}_' \
                                            f'{occlusion_size[1]}_occlusionColor_{occlusion_color}_outputDim_' \
                                            f'{output_dim}_blockSize_{block_size[0]}_{block_size[1]}.json'
        with open(result_filepath, 'w') as f:
            json.dump(results, f)
            f.write('\n')
            f.flush()
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    model_seq = args.model
    dataset = args.dataset

    global model_name
    global input_size
    global output_dim
    global channel
    global mean, std

    if dataset == 'mnist':
        if model_seq == 1:
            model_name = "fnn_model_mnist_1.onnx"
        elif model_seq == 2:
            model_name = "fnn_model_mnist_2.onnx"
        elif model_seq == 3:
            model_name = "fnn_model_mnist_3.onnx"
        channel = 1
        input_size = (28, 28)
        output_dim = 10
        mean = np.array([0.1307])
        std = np.array([0.3081])
        block_size = (7, 7)
    elif dataset == 'gtsrb':
        if model_seq == 1:
            model_name = "fnn_model_gtsrb_small_1_different_class.onnx"
        elif model_seq == 2:
            model_name = "fnn_model_gtsrb_small_2.onnx"
        elif model_seq == 3:
            model_name = "fnn_model_gtsrb_small_3.onnx"
        channel = 3
        input_size = (32, 32)
        output_dim = 7
        mean, std = np.array([0.3337, 0.3064, 0.3171]), np.array([0.2672, 0.2564, 0.2629])
        block_size = (8, 8)
    else:
        raise Exception("not supported dataset")

    # img_loader = get_test_images_loader(input_size, output_dim=output_dim, classes=[1, 2, 3, 4, 5, 7, 8])
    img_loader = get_test_images_loader(input_size, output_dim=output_dim, classes=[1, 2, 3, 4, 5, 7, 8]) if dataset == 'gtsrb' else get_mnist_test_images_loader()
    iterable_img_loader = iter(img_loader)

    # if not use_marabou:
    #     for i in range(batch_num):
    #         image, label = next(iterable_img_loader)
    #         image = image.numpy()
    #         label = label.item()
    #         robust, adv_num, sample_num, total_time = traverse_occlusion_with_fixed_size_by_onnx(image, label,
    #                                                                                              occlusion_size,
    #                                                                                              occlusion_color)
    #         print("total time: ", total_time)
    #         print("robust: ", robust)
    #         print("adv num: ", adv_num)
    #         print("sample num: ", sample_num)
    #     exit(0)

    results = []
    for i in range(batch_num):
        start_time = time.monotonic()
        print(f'image {i}', flush=True)
        image, label = iterable_img_loader.next()
        image = image.numpy()
        print("image shape")
        print(image.shape, flush=True)
        label = label.item()
        isRobust = True
        # constraints_calculation_time = -1.0
        # verify_time = -1.0
        predicted_label = -1
        results_batch = []
        adversarial_example = None
        adv_example_list = None

        for j in range(1, 11):
            occlusion_size = (j, j)
            vals, constraints_calculation_time, verify_time = verify_occlusion_by_dividing(image, label,
                                                                                             occlusion_size,
                                                                                             occlusion_color,
                                                                                             block_size)
            results_batch.append(
             {'occlusion_size': occlusion_size, 'vals': vals[0], 'constraints_calculation_time': constraints_calculation_time,
              'verify_time': verify_time, 'robust': False if vals[0] == 'sat' else True})
            if vals[0] == 'sat':
             adversarial_example = vals[1]
             # unpack adversarial example to a list
             # adversarial example is a dict{int, float}
             # key is the index of the variable in the network
             # value is the value of the variable
             adv_example_list = [adversarial_example[i] for i in range(channel * input_size[0] * input_size[1])]
            # break
        # for target_label in range(output_dim):
        #     if target_label == label:
        #         continue
        #     vals, constraints_calculation_time, verify_time = verify_occlusion_with_fixed_size(image, target_label, occlusion_size, occlusion_color)
        #     results_batch.append(
        #         {'vals': vals[0], 'constraints_calculation_time': constraints_calculation_time, 'verify_time': verify_time,
        #          'target_label': target_label})
        #     if vals[0] == 'sat':
        #         adversarial_example = vals[1]
        #         # unpack adversarial example to a list
        #         # adversarial_example is a dict{int, float}
        #         # key is the index of the variable in the network
        #         # value is the value of the variable
        #         adv_example_list = [adversarial_example[i] for i in range(channel * input_size[0] * input_size[1])]
        #         predicted_label = target_label
        #         isRobust = False
        #         break
        total_time = time.monotonic() - start_time

        print(f"result for {i}")
        print("total time: ", total_time)
        print(results_batch, flush=True)
        results.append(
            {'total_verify_time': total_time,
             'true_label': label, 'detail': results_batch})

    # save results to file
    result_filepath = result_file_dir + f'{model_name}_batchNum_{batch_num}_occlusionColor_{occlusion_color}_outputDim_{output_dim}_{timestamp}.json'
    with open(result_filepath, 'w') as f:
        json.dump(results, f)
        f.write('\n')
        f.flush()
