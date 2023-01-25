# -*- coding: utf-8 -*-
# created by makise, 2023/1/3
import argparse


class ArgumentParser:
    def __init__(self, dataset, type):
        self.dataset = dataset
        self.type = type

    def parse_args(self):
        if self.type == 'uniform':
            return self.__parse_uniform()
        elif self.type == 'multiform':
            return self.__parse_multiform()
        else:
            raise ValueError('type must be either uniform or multiform')


    def log_args(self):
        print("=" * 20)
        print('Arguments:')
        for arg in vars(self.args):
            print(arg, getattr(self.args, arg))
        print("=" * 20, flush=True)


    def __parse_uniform(self):
        """
        Optional Command Line Arguments:
        --model: small / medium / large
        --color: 0-1
        --testNum: number of test images
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='small', help='small / medium / large')
        if self.dataset == 'gtsrb':
            parser.add_argument('--color', type=float, nargs='+', default=[0.0, 0.0, 0.0], help='3 values in [0,1]')
        elif self.dataset == 'mnist':
            parser.add_argument('--color', type=float, default=0.0, help='value in [0,1]')
        else:
            raise ValueError('Invalid dataset name')
        parser.add_argument('--testNum', type=int, default=10, help='number of test images')
        args = parser.parse_args()
        self.args = args
        return args

    def __parse_multiform(self):
        """
        Optional Command Line Arguments:
        --model: small / medium / large
        --testNum: number of test images
        --epsilon: 0.05 / 0.1 / 0.2 / 0.3 / 0.4
        --size: size of occlusion
        --sort: whether using label sorting, default is True
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='small', help='small / medium / large')
        parser.add_argument('--testNum', type=int, default=30, help='number of test images')
        parser.add_argument('--epsilon', type=float, default=0.4, help='0.05 / 0.1 / 0.2 / 0.3 / 0.4')
        parser.add_argument('--size', type=int, default=5, help='size of occlusion, 2 / 5')
        parser.add_argument('--sort', type=int, default=1,
                            help='whether using label sorting, default is True')
        parser.add_argument('--workers', type=int, default=8, help='number of workers')
        parser.add_argument('--split', type=int, default=4, help='input splitting setting')
        args = parser.parse_args()
        self.args = args
        return args