# -*- coding: utf-8 -*-
# created by makise, 2022/9/5
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms, datasets

from occlusion_layer.occlusion_layer_v3 import OcclusionLayer as OcclusionLayerV3
from occlusion_layer.occlusion_layer_v4 import OcclusionLayer as OcclusionLayerV4


def save_uniform_counterexample(image, label, idx, size, input, color, dir):
    image = image.reshape(1, 28, 28)
    occlusion_layer_v3 = OcclusionLayerV3(image, None, True)
    input_tensor = torch.tensor([*input, color])
    occluded_image = occlusion_layer_v3(input_tensor)
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(28, 28)
    occluded_image = np.clip(occluded_image, 0, 1)
    plt.imsave(dir + 'uniform_mnist_image_{}_label_{}_size_{}_color_{}.png'.format(idx, label, size, color), occluded_image, cmap='gray')


def save_multiform_counterexample(image, label, idx, input, epsilons, dir):
    image = image.reshape(1, 28, 28)
    occlusion_layer_v4 = OcclusionLayerV4(image, None, dataset='mnist', shrink=False)
    input_tensor = torch.tensor(input)
    epsilons = torch.tensor(epsilons)
    occluded_image = occlusion_layer_v4(input_tensor, epsilons)
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(28, 28)
    occluded_image = np.clip(occluded_image, 0, 1)
    plt.imsave(dir + 'multiform_mnist_image_{}_label_{}.png'.format(idx, label), occluded_image, cmap='gray')


def show_occluded_image_v4(image, input, epsilons):
    image = image.reshape(1, 28, 28)
    occlusion_layer_v4 = OcclusionLayerV4(image, None, False)
    input_tensor = torch.tensor(input)
    epsilons = torch.tensor(epsilons)
    occluded_image = occlusion_layer_v4(input_tensor, epsilons)
    # convert torch into numpy array
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(28, 28)
    # clip occluded_image into [0, 1] with numpy
    occluded_image = np.clip(occluded_image, 0, 1)
    # set image type as np.float32
    image = image.numpy()
    image = image.reshape(28, 28)
    # show origin and occluded image
    plt.subplot(1, 2, 1)
    plt.imshow(occluded_image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.show()


def show_occluded_image_v3(image, input, color):
    # image = image.numpy()
    image = image.reshape(1, 28, 28)

    occlusion_layer_v3 = OcclusionLayerV3(image, None, True)
    input_tensor = torch.tensor([*input, color])
    occluded_image = occlusion_layer_v3(input_tensor)
    occluded_image = occluded_image.numpy()
    occluded_image = occluded_image.reshape(28, 28)
    occluded_image = np.clip(occluded_image, 0, 1)

    plt.imshow(occluded_image, cmap='gray')
    plt.show()


def get_sample_image(idx):
    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_on_loader = iter(test_loader)

    image, label = next(iter_on_loader)
    # get image at index idx
    for i in range(idx):
        image, label = next(iter_on_loader)
    return image, label.item()


def show_mnist_data_in_grid():
    test_loader = data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False)
    iter_on_loader = iter(test_loader)

    for i in range(100):
        image, label = next(iter_on_loader)
        image = image.numpy()
        image = image.reshape(28, 28)
        plt.subplot(10, 10, i + 1)
        # remove all ticks
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':

    """
    Generate counterexample according to the result.json file output by OccRob
    Some required parameters are:
    --result_dir: the directory of result.json
    --type: uniform / multiform
    --mode: one / all
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True,
                        help='the relative directory of result.json file from the generate_counterexample.py file')
    parser.add_argument('--type', type=str, default='uniform', help='uniform / multiform')
    parser.add_argument('--mode', type=str, default='one', help='one / all, one means generate one counterexample for each image')
    args = parser.parse_args()

    # print all args in a nice format
    print(json.dumps(vars(args), indent=4, sort_keys=True))

    # check whether result.json exists
    result_file = args.result_dir + '/result.json'
    try:
        with open(result_file) as f:
            result = json.load(f)
    except FileNotFoundError:
        print('result.json not found in ' + result_file)
        exit(1)
    # check the result_dir match with type
    if args.type == 'uniform' and 'uniform' not in args.result_dir or args.type == 'multiform' and 'multiform' not in args.result_dir:
        print('occlusion type does not match with result_dir')
        exit(1)

    counterexample_dir = args.result_dir + '/counterexample/'
    if not os.path.exists(counterexample_dir):
        os.makedirs(counterexample_dir)

    if args.type == 'multiform':
        for idx, r in enumerate(result):
            if r['robust'] != False:
                continue
            adversarial_example = r['adversarial_example']
            layer_input = [adversarial_example['a'], adversarial_example['size_a'], adversarial_example['b'], adversarial_example['size_b']]
            layer_epsilons = adversarial_example['epsilons']
            image, label = get_sample_image(idx)
            save_multiform_counterexample(image, label, idx, layer_input, layer_epsilons, counterexample_dir)
    elif args.type == 'uniform':
        for idx, r in enumerate(result):
            assert len(r['robusts']) == len(r['adversarial_examples'])
            image, label = get_sample_image(idx)
            for i, robustness in enumerate(r['robusts']):
                if robustness == True:
                    continue
                adversarial_example = r['adversarial_examples'][i]
                layer_input = [adversarial_example['a'], adversarial_example['size_a'], adversarial_example['b'], adversarial_example['size_b']]
                layer_color = adversarial_example['color']
                save_uniform_counterexample(image, label, idx, i, layer_input, layer_color, counterexample_dir)
                if args.mode == 'one':
                    break
    else:
        print('occlusion type is not supported')
        exit(1)


    # i = 4
    # image, label = get_sample_image(17)
    # print("label: ", label.item())
    # v4_layer_input = []
    # epsilons = []
    # read json object from ./result4.json
    # with open('./result4_fnn2_0.4_sort_1_size2_for_example.json', 'r') as f:
    #     verification_results = json.load(f)
    #     # only consider the image #4
    #     for verification_result in verification_results:
    #         if verification_result['robust'] != False:
    #             continue
    #         adversarial_example = verification_result['adversarial_example']
    #         layer_input = [adversarial_example['a'], adversarial_example['size_a'], adversarial_example['b'], adversarial_example['size_b']]
    #         layer_epsilons = adversarial_example['epsilons']
    #         v4_layer_input.append(layer_input)
    #         epsilons.append(layer_epsilons)
    #
    # images = [0, 1, 6, 10, 17, 26]
    # # for i in range(len(v4_layer_input)):
    # i = 1
    # image, label = get_sample_image(images[i])
    # print("layer input: ", v4_layer_input[i])
    # show_occluded_image_v4(image, v4_layer_input[i], epsilons[i])

    # v3_layer_inputs = [
    #     [8.0, 3.0, 10.0, 3.0],
    #     [11.0, 3.0, 9.999999999999996, 3.0],
    #     [11.0, 3.0, 11.4, 3.0],
    #     [10.0, 3.0, 13.6, 3.0],
    #     [13.0, 3.0, 10.99999999999999, 3.0]
    # ]
    # colors = [0, 0.2, 0.4, 0.6000000000000001, 0.8]
    # image, label = get_sample_image(12)
    # print("label: ", label.item())
    # show_occluded_image_v3(image, [14.0, 4.0, 16.0, 4.0], 0)
    # show_gtsrb_data_in_grid()
    # show_mnist_data_in_grid()
