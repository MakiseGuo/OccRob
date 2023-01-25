# -*- coding: utf-8 -*-
# created by makise, 2022/8/4

import concurrent.futures
import time

import pebble


# experiment 2
def determine_robustness(size, labels, model, task):
    colors = [0, 0.2, 0.4, 0.6, 0.8, 1]
    robusts = []
    color_times = []
    for c in range(1):
        color_start_time = time.monotonic()
        position_range = (1, 32 - size + 1)
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [(step, 2 * step), (2 * step, 3 * step), (1, step), (3 * step, position_range[1])]
        robust = True
        for label in labels:
            start_time = time.monotonic()
            if not robust:
                break
            with pebble.ProcessPool(1) as pool:
                color = (0.1, 0.9)
                for i in range(4):
                    a = position[i]
                    b = position[i]
                    size_a = (size, size)
                    size_b = (size, size)
                    future = pool.schedule(task, (model, label, a, b, size_a, size_b, color))
                    try:
                        print('label {}, a&b {}, size_a&b {}, color {}'.format(label, a, size_a, color), flush=True)
                        task_start_time = time.monotonic()
                        result = future.result(timeout=60)
                        print("task end in ", time.monotonic() - task_start_time)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        print("timeout", flush=True)
                        result = 'unsat'  # To keep format with return value of task
                    if result == 'sat':
                        robust = False
                        break
            print("label {} end in {}".format(label, time.monotonic() - start_time))
        robusts.append(robust)
        color_times.append(time.monotonic() - color_start_time)
    return robusts, color_times


def determine_robustness_color_fixed(sizes, labels, model, color, task):
    robusts = []
    size_times = []
    adversarial_examples = []
    timeout_count = 0
    total_num = 0
    for size in range(sizes[0], sizes[1] + 1):
        print("size {}x{} starts:".format(size, size))
        adversarial_example = {}
        size_start_time = time.monotonic()
        position_range = (1, 32 - size + 1)
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [(step, 2 * step), (2 * step, 3 * step), (1, step), (3 * step, position_range[1])]
        robust = True
        for label in labels:
            print("label {} starts:".format(label))
            if not robust:
                break
            with pebble.ProcessPool(1) as pool:
                for i in range(4):
                    total_num += 1
                    a = position[i]
                    b = position[i]
                    size_a = (size, size)
                    size_b = (size, size)
                    future = pool.schedule(task, (model, label, a, b, size_a, size_b, (color, color)))
                    try:
                        result, result_input = future.result(timeout=60)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        print("timeout", flush=True)
                        timeout_count += 1
                        result = 'unsat'  # To keep format with return value of task
                    if result == 'sat':
                        robust = False
                        adversarial_example['a'] = result_input[0]
                        adversarial_example['size_a'] = result_input[1]
                        adversarial_example['b'] = result_input[2]
                        adversarial_example['size_b'] = result_input[3]
                        adversarial_example['color'] = (result_input[4], result_input[5], result_input[6])
                        break
            print("label {} ends".format(label))
        robusts.append(robust)
        size_times.append(time.monotonic() - size_start_time)
        adversarial_examples.append(adversarial_example)
        print("size {}x{} ends".format(size, size))
    return robusts, adversarial_examples, size_times, timeout_count / total_num


def determine_robustness_with_epsilon(size, labels, epsilon, model, task, workers, split):
    task_start_time = time.monotonic()
    robust = True
    adversarial_example = {}
    timeout_count = 0
    total_num = 0
    position_range = (1, 32 - size[0] + 1)
    if split == 4:
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [(step, 2 * step), (2 * step, 3 * step), (1, step), (3 * step, position_range[1])]
    else:
        step = (position_range[1] - position_range[0] + 1) // split
        position = [(max(1, i * step), min((i + 1) * step, position_range[1])) for i in range(split)]
    for label in labels:
        if not robust:
            break
        start_time = time.monotonic()
        params_position_a = []
        params_position_b = []
        for i in range(split):
            for j in range(split):
                params_position_a.append(position[i])
                params_position_b.append(position[j])

        with pebble.ProcessPool(workers) as pool:
            future = pool.map(task, [model] * (split * split), [label] * (split * split), params_position_a,
                              params_position_b, [size] * (split * split),
                              [size] * (split * split), [epsilon] * (split * split), timeout=60)
            iterator = future.result()
            while True:
                try:
                    total_num += 1
                    result, result_input = next(iterator)
                    print("result is ", result, flush=True)
                    if result == 'sat':
                        robust = False
                        adversarial_example['a'] = result_input[0]
                        adversarial_example['size_a'] = result_input[1]
                        adversarial_example['b'] = result_input[2]
                        adversarial_example['size_b'] = result_input[3]
                        adversarial_example['epsilons'] = [result_input[i] for i in range(4, 4 + (32 * 32))]
                        print("verification exit in idx at label {}".format(label), flush=True)
                        future.cancel()
                except StopIteration:
                    print("iterator ends")
                    break
                except concurrent.futures.TimeoutError as error:
                    print("timeout")
                    print(error.args, flush=True)
                    timeout_count += 1
                except Exception:
                    print("error", flush=True)
        print("label {} end in {}".format(label, time.monotonic() - start_time), flush=True)
    return robust, adversarial_example, timeout_count / total_num, time.monotonic() - task_start_time
