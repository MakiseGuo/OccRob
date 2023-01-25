# -*- coding: utf-8 -*-
# created by makise, 2022/2/18

import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None, classes=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            classes (list, optional): List of classes to be loaded.
        """
        self.root_dir = root_dir

        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        csv_data = pd.read_csv(csv_file_path)
        # cut the dataframe to the classes we want to load
        # get the second column of the csv file and check it against the classes
        if classes is not None:
            csv_data = csv_data[csv_data.iloc[:, 1].isin(classes)]

        self.csv_data = csv_data

        self.transform = transform

        self.classes = classes

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)
        # label is the idx of classId in classes
        label = self.classes.index(classId)
        return img, label