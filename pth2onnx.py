# -*- coding: utf-8 -*-
# created by makise, 2022/2/22

"""
convert pth format network to onnx format network
"""

import torch
from torch.utils import data
from torchvision import transforms
from gtsrb.gtsrb_dataset import GTSRB
from gtsrb.cnn_model_small import SmallCNNModel
from gtsrb.fnn_model_1 import SmallDNNModel
from gtsrb.fnn_model_2 import SmallDNNModel2
from gtsrb.cnn_model_small_2 import SmallCNNModel2
from gtsrb.fnn_model_3 import SmallDNNModel3
from mnist.fnn_model_1 import FNNModel1 as MNISTFNNModel1
from mnist.fnn_model_3 import FNNModel1 as MNISTFNNModel3
from mnist.fnn_model_2 import FNNModel1 as MNISTFNNModel2
import onnx
import onnxruntime

# define some global parameters for exporting the pytorch model
model_name = 'mnist_fnn_2'
use_device = 'cpu'
input_size = (28, 28)
channel_num = 1
output_dim = 10
batch_size = 1
model_path = 'model/fnn_model_mnist_2.pth'
onnx_model_path = 'model/fnn_model_gtsrb_small_3.onnx'  # only used in testing
model_save_dir = 'model/'
only_export = True
only_test = False


def initialize_model(model_name):
    if model_name == 'gtsrb_cnn_small':
        model = SmallCNNModel()
    elif model_name == 'gtsrb_fnn_1':
        model = SmallDNNModel()
    elif model_name == 'gtsrb_fnn_2':
        model = SmallDNNModel2()
    elif model_name == 'gtsrb_cnn_small_2':
        model = SmallCNNModel2()
    elif model_name == 'gtsrb_fnn_3':
        model = SmallDNNModel3()
    elif model_name == 'mnist_fnn_1':
        model = MNISTFNNModel1()
    elif model_name == 'mnist_fnn_3':
        model = MNISTFNNModel3()
    elif model_name == 'mnist_fnn_2':
        model = MNISTFNNModel2()
    else:
        raise ValueError('model name is not defined')

    return model


def initialize_device():
    device = torch.device('cuda' if torch.cuda.is_available() and use_device == 'gpu' else 'cpu')
    return device


def export_model_2_onnx(model, model_name, device, input_size, channel_num, batch_size, model_path, model_save_dir):
    # load state dict of model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    # export the model to onnx format
    dummy_input = torch.randn(batch_size, channel_num, input_size[0], input_size[1])
    onnx_model_filename = model_path.split('/')[-1].split('.')[0] + '.onnx'
    torch.onnx.export(model, dummy_input, model_save_dir + onnx_model_filename, verbose=True)


# test onnx model using one batch size test samples
def test_model_onnx(onnx_model_path, input_size, channel_num, output_dim, batch_size):
    # define the same data transform as when the model is trained
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629])
    ])
    # define the test data
    if output_dim != 43:
        test_data = GTSRB(root_dir='data/', train=False, transform=data_transform, classes=[1, 2, 3, 4, 5, 7, 8])
    else:
        test_data = GTSRB(root_dir='data/', train=False, transform=data_transform)
    # create data loader for evaluating
    test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    samples, labels = iter(test_loader).next()
    print("samples shape: ", samples.shape)
    print("labels shape: ", labels.shape)
    acc = 0
    for i in range(100):
        # create the onnxruntime session
        ort_session = onnxruntime.InferenceSession(onnx_model_path)
        # create the input tensor
        input_name = ort_session.get_inputs()[0].name
        input_tensor = samples.numpy()
        input_tensor = input_tensor.reshape(batch_size, channel_num, input_size[0], input_size[1])
        # run the model
        output_tensor = ort_session.run(None, {input_name: input_tensor})  # the torch_out is 1 * batch_size * output_dim
        output_tensor = torch.tensor(output_tensor[0])
        _, predicted = torch.max(output_tensor, 1)
        acc += (predicted == labels).sum().item() / batch_size
    print(f'accuracy: {acc}%')


if __name__ == '__main__':
    if only_test:
        test_model_onnx(onnx_model_path, input_size, channel_num, output_dim, batch_size)
    else:
        model = initialize_model(model_name)
        device = initialize_device()
        export_model_2_onnx(model, model_name, device, input_size, channel_num, batch_size, model_path, model_save_dir)
        if not only_export:
            test_model_onnx(onnx_model_path, input_size, channel_num, output_dim, batch_size)
