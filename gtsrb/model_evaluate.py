# -*- coding: utf-8 -*-
# created by makise, 2022/2/20
import csv

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms

from cnn_model_1 import AlexnetTSR
from cnn_model_1 import OUTPUT_DIM
from gtsrb_dataset import GTSRB
from torch.utils.data import DataLoader

from transformer import OcclusionTransform, MulOcclusionTransform


# show and save image
def imshow_and_save(img, save_name):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    # unnormalize the image, it's working, and the warning can be ignored
    npimg = npimg * np.array((0.2672, 0.2564, 0.2629)) + np.array((0.3337, 0.3064, 0.3171))
    plt.imshow(npimg)
    # save the image to experiment/occlusion_effect folder concatenate the image name with the save_path
    plt.savefig(f'../experiment/occlusion_effect/{save_name}.png')
    plt.show()



# load the model from file
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexnetTSR()
model.load_state_dict(torch.load('../model/cnn_model_gtsrb.pth', map_location=device))
model = model.to(device)

# define some hyperparameters
output_dim = OUTPUT_DIM
batch_size = 64

# define customized data_transform
data_transform_occlusion_0_0_0 = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
])
data_transform_occlusion_30_30_1 = transforms.Compose([
    transforms.Resize((112, 112)),
    OcclusionTransform(occlusion_height=30, occlusion_width=30),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
data_transform_occlusion_30_30_2 = transforms.Compose([
    transforms.Resize((112, 112)),
    MulOcclusionTransform(occlusion_height=30, occlusion_width=30, occlusion_num=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
data_transform_occlusion_10_10_1 = transforms.Compose([
    transforms.Resize((112, 112)),
    OcclusionTransform(occlusion_height=10, occlusion_width=10),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
data_transform_occlusion_10_10_2 = transforms.Compose([
    transforms.Resize((112, 112)),
    MulOcclusionTransform(occlusion_height=10, occlusion_width=10, occlusion_num=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
data_transform_occlusion_10_10_3 = transforms.Compose([
    transforms.Resize((112, 112)),
    MulOcclusionTransform(occlusion_height=10, occlusion_width=10, occlusion_num=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
data_transform_occlusion_20_20_1 = transforms.Compose([
    transforms.Resize((112, 112)),
    OcclusionTransform(occlusion_height=20, occlusion_width=20),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
data_transform_occlusion_20_20_2 = transforms.Compose([
    transforms.Resize((112, 112)),
    MulOcclusionTransform(occlusion_height=20, occlusion_width=20, occlusion_num=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
data_transform_occlusion_20_20_3 = transforms.Compose([
    transforms.Resize((112, 112)),
    MulOcclusionTransform(occlusion_height=20, occlusion_width=20, occlusion_num=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
data_transforms = {
    'occlusion_0_0_0': data_transform_occlusion_0_0_0,
    'occlusion_30_30_1': data_transform_occlusion_30_30_1,
    'occlusion_30_30_2': data_transform_occlusion_30_30_2,
    'occlusion_10_10_1': data_transform_occlusion_10_10_1,
    'occlusion_10_10_2': data_transform_occlusion_10_10_2,
    'occlusion_10_10_3': data_transform_occlusion_10_10_3,
    'occlusion_20_20_1': data_transform_occlusion_20_20_1,
    'occlusion_20_20_2': data_transform_occlusion_20_20_2,
    'occlusion_20_20_3': data_transform_occlusion_20_20_3
}

results = []
# iterate over the data_transforms
for key in data_transforms:
    data_transform = data_transforms[key]
    # print some log information
    print("current data transform: ", key)

    # define the test dataset
    test_data = GTSRB(root_dir='../data', train=False, transform=data_transform)

    # define the test dataloader
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # show some samples in test dataset
    samples, labels = iter(test_loader).next()
    print(samples.shape)
    # use the data_transform name as the save_path
    imshow_and_save(torchvision.utils.make_grid(samples), key)

    # evaluate the loaded model
    with torch.no_grad():
        model.eval()
        result_dict = {}
        # extract last three terms from the data_transform name
        occlusion_height, occlusion_width, occlusion_num = key.split('_')[-3:]
        occlusion_height, occlusion_width, occlusion_num = int(occlusion_height), int(occlusion_width), int(
            occlusion_num)
        result_dict['occlusion_height'] = occlusion_height
        result_dict['occlusion_width'] = occlusion_width
        result_dict['occlusion_num'] = occlusion_num
        result_dict['image_height'] = 112
        result_dict['image_width'] = 112

        correct = 0
        total = 0
        class_correct = [0] * output_dim
        class_total = [0] * output_dim

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                # in the last batch, the batch size may be smaller than batch_size
                if i < len(labels):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        acc = 100.0 * correct / total
        result_dict['accuracy'] = acc
        print(f'Accuracy of the network: {acc} %')

        for i in range(output_dim):
            acc = 100.0 * class_correct[i] / class_total[i]
            result_dict['class_accuracy_' + str(i)] = acc
            print(f'Accuracy of {i} class: {acc} %')

        results.append(result_dict)

# save the results into a csv file
with open('../experiment/occlusion_effect/results.csv', 'w') as csvfile:
    fieldnames = ['occlusion_height', 'occlusion_width', 'occlusion_num', 'image_height', 'image_width', 'accuracy',
                  'accuracy_change_rate',
                  'class_accuracy_0', 'class_accuracy_1', 'class_accuracy_2', 'class_accuracy_3', 'class_accuracy_4',
                  'class_accuracy_5', 'class_accuracy_6', 'class_accuracy_7', 'class_accuracy_8', 'class_accuracy_9',
                  'class_accuracy_10', 'class_accuracy_11', 'class_accuracy_12', 'class_accuracy_13',
                  'class_accuracy_14', 'class_accuracy_15', 'class_accuracy_16', 'class_accuracy_17',
                  'class_accuracy_18', 'class_accuracy_19', 'class_accuracy_20', 'class_accuracy_21',
                  'class_accuracy_22', 'class_accuracy_23', 'class_accuracy_24', 'class_accuracy_25',
                  'class_accuracy_26', 'class_accuracy_27', 'class_accuracy_28', 'class_accuracy_29',
                  'class_accuracy_30', 'class_accuracy_31', 'class_accuracy_32', 'class_accuracy_33',
                  'class_accuracy_34', 'class_accuracy_35', 'class_accuracy_36', 'class_accuracy_37',
                  'class_accuracy_38', 'class_accuracy_39', 'class_accuracy_40', 'class_accuracy_41',
                  'class_accuracy_42',
                  'class_accuracy_change_rate_0', 'class_accuracy_change_rate_1', 'class_accuracy_change_rate_2',
                  'class_accuracy_change_rate_3', 'class_accuracy_change_rate_4', 'class_accuracy_change_rate_5',
                  'class_accuracy_change_rate_6', 'class_accuracy_change_rate_7', 'class_accuracy_change_rate_8',
                  'class_accuracy_change_rate_9', 'class_accuracy_change_rate_10', 'class_accuracy_change_rate_11',
                  'class_accuracy_change_rate_12', 'class_accuracy_change_rate_13', 'class_accuracy_change_rate_14',
                  'class_accuracy_change_rate_15', 'class_accuracy_change_rate_16', 'class_accuracy_change_rate_17',
                  'class_accuracy_change_rate_18', 'class_accuracy_change_rate_19', 'class_accuracy_change_rate_20',
                  'class_accuracy_change_rate_21', 'class_accuracy_change_rate_22', 'class_accuracy_change_rate_23',
                  'class_accuracy_change_rate_24', 'class_accuracy_change_rate_25', 'class_accuracy_change_rate_26',
                  'class_accuracy_change_rate_27', 'class_accuracy_change_rate_28', 'class_accuracy_change_rate_29',
                  'class_accuracy_change_rate_30', 'class_accuracy_change_rate_31', 'class_accuracy_change_rate_32',
                  'class_accuracy_change_rate_33', 'class_accuracy_change_rate_34', 'class_accuracy_change_rate_35',
                  'class_accuracy_change_rate_36', 'class_accuracy_change_rate_37', 'class_accuracy_change_rate_38',
                  'class_accuracy_change_rate_39', 'class_accuracy_change_rate_40', 'class_accuracy_change_rate_41',
                  'class_accuracy_change_rate_42']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        # save the accuracy_change_rate as a percentage of how much the accuracy changed compared to data_transforms_0_0_0 result
        result['accuracy_change_rate'] = (result['accuracy'] - results[0]['accuracy']) / results[0]['accuracy'] * 100
        # save the class_accuracy_change_rate as a percentage of how much the accuracy changed compared to data_transforms_0_0_0 result
        for i in range(43):
            result['class_accuracy_change_rate_' + str(i)] = (result['class_accuracy_' + str(i)] - results[0][
                'class_accuracy_' + str(i)]) / results[0]['class_accuracy_' + str(i)] * 100

        writer.writerow(result)
