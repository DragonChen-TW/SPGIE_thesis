import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import argparse
from tqdm import tqdm
# 
import sys
sys.path.append('..')
from spdatasets.mnist import MNISTSuperPixelDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset_name', type=str, default='mnist')
args = parser.parse_args('')

dataset_name = args.dataset_name

batch_size = args.batch_size

if dataset_name == 'mnist':
    dataset_cls = MNISTSuperPixelDataset
    ckpt_filename = f'ckpts_{dataset_name}/mnist_e400_test9699.pt'
# elif dataset_name == 'mnist_m':
#     dataset_cls = MNISTMSuperPixelDataset
#     ckpt_filename = f'ckpts_{dataset_name}/mnist_e400_test9699.pt'
# elif dataset_name == 'fashion_mnist':
#     dataset_cls = FASHIONMNISTSuperPixelDataset
# elif dataset_name == 'svhn':
#     dataset_cls = SVHNSuperPixelDataset
# elif dataset_name == 'cifar10':
#     dataset_cls = CIFAR10SuperPixelDataset

path = f'~/data/{dataset_name.upper()}_SUPERPIXEL'

test_dataset = dataset_cls(path, train=False)
# test_dataset = DATASET(path, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

imgs = next(iter(test_loader))
print(imgs)

print('features =', test_dataset.num_features)
print('classes =', test_dataset.num_classes)
hidden_dim = 20
print('hidden_dim =', hidden_dim)

from jit_drn_model import DynamicReductionNetworkJit

input_dim = test_dataset.num_features
hidden_dim = 20
output_dim = test_dataset.num_classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drn = DynamicReductionNetworkJit(input_dim=input_dim, hidden_dim=hidden_dim,
                                           output_dim=output_dim,
                                           k=4, aggr='add',
                                           agg_layers=2, mp_layers=2, in_layers=3, out_layers=3,
                                           graph_features=3,
                                           )
    def forward(self, data):
        logits = self.drn(data.x, data.batch, None) # TO CHECK
        return F.log_softmax(logits, dim=1)

model = Net()

device = torch.device('cpu')
ckpt = torch.load('../checkpoints/mnist_e400_test9699.pt', map_location=device)
model.load_state_dict(ckpt)

from drn_train import test

test_acc, test_loss = test(model, test_loader, device)

print(f'acc = {test_acc * 100:5.2f}, loss = {test_loss:.3f}')