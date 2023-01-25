# -*- coding: utf-8 -*-
# created by makise, 2022/7/26
import concurrent.futures
import time

import pebble


# experiment 2
def determine_robustness(size, labels, models, task):
    colors = [0, 0.2, 0.4, 0.6, 0.8, 1]
    robusts = []
    color_times = []
    for c in range(5):
        color_start_time = time.monotonic()
        position_range = (1, 28 - size + 1)
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [(step, 2 * step), (2 * step, 3 * step), (1, step), (3 * step, position_range[1])]
        robust = True
        for label in labels:
            start_time = time.monotonic()
            if not robust:
                break
            with pebble.ProcessPool(1) as pool:
                color = (colors[c], colors[c + 1])
                for i in range(4):
                    a = position[i]
                    b = position[i]
                    size_a = (size, size)
                    size_b = (size, size)
                    future = pool.schedule(task, (models[i], label, a, b, size_a, size_b, color), timeout=60)
                    try:
                        print('label {}, a&b {}, size_a&b {}, color {}'.format(label, a, size_a, color), flush=True)
                        task_start_time = time.monotonic()
                        result = future.result()
                        print("task end in ", time.monotonic() - task_start_time)
                    except concurrent.futures.TimeoutError:
                        print("timeout", flush=True)
                        result = 'unsat'  # To keep format with return value of task
                    if result == 'sat':
                        robust = False
                        break
            print("label {} end in {}".format(label, time.monotonic() - start_time))
        robusts.append(robust)
        color_times.append(time.monotonic() - color_start_time)
    return robusts, color_times


# experiment 1
def find_robust_lower_bound(l, u, labels, model, task):
    lower = l
    _, image_height, image_width = (1, 28, 28)
    upper = u
    upper_last_sat = upper
    iter_count = 0
    while upper - lower >= 1:
        iter_count += 1
        position_range = (1, image_height - lower + 1)
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [1, step, 2 * step, 3 * step, position_range[1]]
        robust = True
        for label in labels:
            start_time = time.monotonic()
            if not robust:
                break
            with pebble.ProcessPool(1) as pool:
                for i in range(4):
                    a = (position[i], position[i + 1])
                    b = (position[i], position[i + 1])
                    size_a = (lower, upper)
                    size_b = (lower, upper)
                    future = pool.schedule(task, (model, label, a, b, size_a, size_b, (0, 0)))
                    try:
                        print('iteration {}: label {}, a&b {}, size_a&b {}'.format(iter_count, label, a, size_a),
                              flush=True)
                        task_start_time = time.monotonic()
                        result = future.result(timeout=60)
                        print("task end in ", time.monotonic() - task_start_time)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        print("iteration {} timeout".format(iter_count), flush=True)
                        result = 'unsat'  # To keep format with return value of task
                    if result == 'sat':
                        robust = False
                        break
            print("label {} end in {}".format(label, time.monotonic() - start_time))
        if not robust:
            print("size {} is not robust".format((lower, upper)), flush=True)
            upper_last_sat = upper
            upper = (upper + lower) // 2
        else:
            print("size {} is robust".format((lower, upper)), flush=True)
            lower = upper
            upper = (upper_last_sat + lower) // 2
    return upper


def determine_robustness_color_fixed(sizes, labels, model, color, task):
    robusts = []
    size_times = []
    adversarial_examples = []
    timeout_count = 0
    total_num = 0
    for size in range(sizes[0], sizes[1] + 1):
        adversarial_example = {}
        print("size {}x{} starts:".format(size, size))
        size_start_time = time.monotonic()
        position_range = (1, 28 - size + 1)
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
                        adversarial_example['color'] = result_input[4]
                        break
            print("label {} ends".format(label))
        robusts.append(robust)
        size_times.append(time.monotonic() - size_start_time)
        adversarial_examples.append(adversarial_example)
        print("size {}x{} ends".format(size, size))
    timeout_prop = timeout_count / total_num
    return robusts, adversarial_examples, size_times, timeout_prop


def determine_robustness_with_epsilon(size, labels, epsilon, model, task, workers, split):
    task_start_time = time.monotonic()
    robust = True
    adversarial_example = {}
    timeout_count = 0
    total_num = 0
    position_range = (1, 28 - size[0] + 1)
    if split == 4:
        step = (position_range[1] - position_range[0] + 1) // 4
        position = [(step, 2 * step), (2 * step, 3 * step), (1, step), (3 * step, position_range[1])]
    elif split == 5:
        step = (position_range[1] - position_range[0] + 2) // 5
        position = [(step, 2 * step), (2 * step, 3 * step), (3 * step, 4 * step), (1, step),
                    (4 * step, position_range[1])]
        # position = [(5, 10), (10, 15), (15, 20), (1, 5), (20, 24)]
    else:
        step = (position_range[1] - position_range[0] + 1) // split
        position = [(max(1, i * step), min((i + 1) * step, position_range[1])) for i in range(split)]
    for label in labels:
        if not robust:
            break
        start_time = time.monotonic()
        params_model = []
        params_label = []
        params_position_a = []
        params_position_b = []
        params_size = []
        params_epsilon = []
        for i in range(split):
            for j in range(split):
                params_model.append(model)
                params_label.append(label)
                params_position_a.append(position[i])
                params_position_b.append(position[j])
                params_size.append(size)
                params_epsilon.append(epsilon)

        with pebble.ProcessPool(workers) as pool:
            future = pool.map(task, params_model, params_label, params_position_a, params_position_b, params_size,
                              params_size, params_epsilon, timeout=60)
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
                        adversarial_example['epsilons'] = [result_input[i] for i in range(4, 4 + (28 * 28))]
                        print("verification exit in idx at label".format(label), flush=True)
                        future.cancel()
                except StopIteration:
                    print("iterator ends")
                    break
                except concurrent.futures.TimeoutError as error:
                    print("timeout, ", error.args, flush=True)
                    timeout_count += 1
                except concurrent.futures.CancelledError as error:
                    print("cancelled, ", error.args, flush=True)
                except Exception:
                    print("error", flush=True)
        print("label {} end in {}".format(label, time.monotonic() - start_time), flush=True)
    return robust, adversarial_example, timeout_count / total_num, time.monotonic() - task_start_time