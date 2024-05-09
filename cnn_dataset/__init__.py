from torchvision.datasets import (
    MNIST, FashionMNIST,
    CIFAR10, SVHN
)
from .mnistm import MNISTM

import torch
DATASET_MAPPING = {
    'mnist': MNIST,
    'mnist_m': MNISTM,
    'fashion_mnist': FashionMNIST,
    'cifar10': CIFAR10,
    'svhn': SVHN,
}
def cnndataset_mapping(dataset_name: str) -> torch.data.Dataset:
    return DATASET_MAPPING[dataset_name]

# # MNIST
# # FashionMNIST
# def __init__(
#     self,
#     root: str,
#     train: bool = True,
#     transform: Optional[Callable] = None,
#     target_transform: Optional[Callable] = None,
#     download: bool = False,
# )

# # CIFAR10
# def __init__(
#     self,
#     root: str,
#     train: bool = True,
#     transform: Optional[Callable] = None,
#     target_transform: Optional[Callable] = None,
#     download: bool = False,
# ) -> None:

# # SVHN
# def __init__(
#     self,
#     root: str,
#     split: str = "train",
#     transform: Optional[Callable] = None,
#     target_transform: Optional[Callable] = None,
#     download: bool = False,
# )

# # MNISTM
# def __init__(self, root, mode='train', to_gray=False):
#     pass