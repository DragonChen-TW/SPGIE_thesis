from .mnist import MNISTSuperPixelDataset
from .mnistm import MNISTMSuperPixelDataset
from .fashion_mnist import FASHIONMNISTSuperPixelDataset
from .cifar10 import CIFAR10SuperPixelDataset
from .svhn import SVHNSuperPixelDataset

import torch
DATASET_MAPPING = {
    'mnist': MNISTSuperPixelDataset,
    'mnist_m': MNISTMSuperPixelDataset,
    'fashion_mnist': FASHIONMNISTSuperPixelDataset,
    'cifar10': CIFAR10SuperPixelDataset,
    'svhn': SVHNSuperPixelDataset,
}
def spdataset_mapping(dataset_name: str) -> torch.utils.data.Dataset:
    return DATASET_MAPPING[dataset_name]