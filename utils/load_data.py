"""
Description: A library for data loaders.
"""

import os
import cv2
from skimage import io

import numpy as np
import pandas as pd
import datatable as dt
from collections import Counter
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pandas import read_csv

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms


class Loader(object):
    """ Data loader """

    def __init__(self, config, dataset_name, eval_mode=False, kwargs={}):
        """
        :param dict config: Configuration dictionary.
        :param str dataset_name: Name of the dataset to use.
        :param bool eval_mode: Whether the dataset is used for evaluation. False by default.
        :param dict kwargs: Additional parameters if needed.
        """
        # Get batch size
        batch_size = config["batch_size"]
        # Get config
        self.config = config
        # Set main results directory using database name. Exp:  processed_data/dpp19
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name)
        # Get the datasets
        train_dataset, test_dataset, validation_dataset = self.get_dataset(dataset_name, file_path, eval_mode=eval_mode)
        # Set the loader for training set
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        # Set the loader for test set
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
        # Set the loader for validation set
        self.validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

    def get_dataset(self, dataset_name, file_path, eval_mode=False):
        # Create dictionary for loading functions of datasets.
        # If you add a new dataset, add its corresponding dataset class here in the form 'dataset_name': ClassName
        loader_map = {'default_loader': TabularDataset, 'mnist': TabularDataset}
        # Get dataset. Check if the dataset has a custom class. If not, then assume a tabular data with labels in the first column
        dataset = loader_map[dataset_name] if dataset_name in loader_map.keys() else loader_map['default_loader']
        # Training and Validation datasets
        train_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode='train')
        # Test dataset
        test_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode='test')
        # validation dataset
        validation_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode="validation")
        # Return
        return train_dataset, test_dataset, validation_dataset


class ToTensorNormalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # Assumes that min-max scaling is done when pre-processing the data
        return torch.from_numpy(sample).float()


class TabularDataset(Dataset):
    def __init__(self, config, datadir, dataset_name, mode='train', transform=ToTensorNormalize()):
        """
        Expects two csv files with _tr and _te suffixes for training and test datasets.
        Example: dataset_name_tr.csv, dataset_name_te.csv
        """
        self.config = config
        self.datadir = datadir
        self.dataset_name = dataset_name
        self.mode = mode
        self.data, self.labels = self._load_data()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        cluster = int(self.labels[idx]) if self.mode != "validation" else None
        return sample, cluster

    def _load_data(self):


        training_data_ratio = self.config["training_data_ratio"]
        dataset = MNIST("./data/", download=True)

        with open('./data/MNIST/processed/train.npy', 'rb') as f:
            x_train = np.load(f)
            y_train = np.load(f)
            
        with open('./data/MNIST/processed/test.npy', 'rb') as f:
            x_test = np.load(f)
            y_test = np.load(f)

        x_train = x_train.reshape(-1, 28*28)/255.
        x_test = x_test.reshape(-1, 28*28)/255.
        
        # Divide labeled and validation data
        idx = np.random.permutation(len(y_train))

        # Label data : validation data = training_data_ratio:(1-training_data_ratio)
        tr_idx = idx[:int(len(idx) * training_data_ratio)]
        val_idx = idx[int(len(idx) * training_data_ratio):]

        # validation data
        x_val = x_train[val_idx, :]
        y_val = y_train[val_idx]

        # Labeled data
        x_train = x_train[tr_idx, :]
        y_train = y_train[tr_idx]

        if self.mode =="train":
            data = x_train
            labels = y_train
            # Return features, and labels
            return data, labels  
        elif self.mode == "test":
            data = x_test
            labels = y_test
            # Return features, and labels
            return data, labels  
        elif self.mode == "validation":
            data = x_val
            labels = y_val
        # Return features, and labels
        return data.reshape(-1,28*28), labels    
