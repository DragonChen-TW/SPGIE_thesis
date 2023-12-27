import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import mlflow
# 
from spdataset import (
    MNISTSuperPixelDataset, MNISTMSuperPixelDataset,
    FASHIONMNISTSuperPixelDataset,
    CIFAR10SuperPixelDataset,
    SVHNSuperPixelDataset,
)
from train.jit_drn_model import DynamicReductionNetworkJit
from train.drn_train import train, test
from utils.meter import Meter
from utils.scheduler import CyclicLRWithRestarts

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--hidden_dim', type=int, default=20)
parser.add_argument('--max_epoch', type=int, default=20)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
param = vars(args)

dataset_name = args.dataset_name
batch_size = args.batch_size
num_superpixel = '75'

param.update({
    'num_superpixel': int(num_superpixel),
})

ckpt_path = f'ckpts/ckpt_{dataset_name}'
os.makedirs(ckpt_path, exist_ok=True)

print('dataset', dataset_name, num_superpixel)

if dataset_name == 'mnist':
    dataset_cls = MNISTSuperPixelDataset
elif dataset_name == 'mnist_m':
    dataset_cls = MNISTMSuperPixelDataset
elif dataset_name == 'fashion_mnist':
    dataset_cls = FASHIONMNISTSuperPixelDataset
elif dataset_name == 'cifar10':
    dataset_cls = CIFAR10SuperPixelDataset
elif dataset_name == 'svhn':
    dataset_cls = SVHNSuperPixelDataset

path = f'~/data/{dataset_name.upper()}_SUPERPIXEL'
if num_superpixel != '75':
    path += f'_{num_superpixel}'

transform = None
train_dataset = dataset_cls(path, train=True, transform=transform)
test_dataset = dataset_cls(path, train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

epoch_size = len(train_dataset)
print('Epoch', epoch_size, 'Batch size', batch_size)
print('features ->', train_dataset.num_features)
print('classes ->',train_dataset.num_classes)

input_dim = train_dataset.num_features

input_dim = test_dataset.num_features
hidden_dim = args.hidden_dim
output_dim = test_dataset.num_classes
print('hidden_dim = {}'.format(hidden_dim))

drn_k = 4
aggr = 'add'
pool = 'max'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drn = DynamicReductionNetworkJit(
            input_dim=input_dim, hidden_dim=hidden_dim,
            output_dim=output_dim,
            k=drn_k, aggr=aggr,
            pool=pool,
            agg_layers=2, mp_layers=2, in_layers=3, out_layers=3,
            graph_features=train_dataset.num_features,
        )

    def forward(self, data):
        logits = self.drn(data.x, data.batch, None) # TO CHECK
        return F.log_softmax(logits, dim=1)

param.update({
    'drn_k': drn_k,
    'hidden_dim': hidden_dim,
    'aggr': aggr,
    'pool': pool,
})

def print_model_summary(model):
    """Override as needed"""
    print(
        'Model: \n%s\nParameters: %i' %
        (model, sum(p.numel() for p in model.parameters()))
    )

def save_model(model, ckpt):
    torch.save(model.state_dict(), f'{ckpt_path}/{ckpt}')

def load_model(model, ckpt):
    model.load_state_dict(torch.load(f'{ckpt_path}/{ckpt}', map_location=device))

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.2, policy="cosine")

print_model_summary(model)

max_epoch = args.max_epoch
resume_training = False
# resume_file = 'e50.pt'
# if resume_training:
#     start_epoch = int(resume_file.replace('.pt', '')[1:])
#     ckpt = torch.load(f'{CKPT_PATH}/{resume_file}')
#     model.load_state_dict(ckpt)
# else:
#     start_epoch = 1
start_epoch = 1

param.update({
    'max_epoch': max_epoch,
})

mlflow.log_params(param)

import time
import matplotlib.pyplot as plt

best_acc = 0
m = Meter()

# ========== Training loop ==========
t = time.time()
for epoch in range(start_epoch, max_epoch + 1):
    train_acc, train_loss = train(epoch, model, train_loader, optimizer, scheduler, device)
    test_acc, test_loss = test(model, test_loader, device)
    m.update(train_loss, train_acc, test_loss, test_acc)
    print('Epoch: {:02d}, Train: {:.4f} Test: {:.4f}'.format(epoch, train_acc, test_acc))

    mlflow.log_metrics({
        'train_acc': train_acc,
        'train_loss': train_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
    }, step=epoch - 1)

    if epoch % 5 == 0:
        save_model(model, f'e{epoch:02}.pt')
    if test_acc > best_acc:
        print('best in epoch', epoch)
        print('test acc', test_acc)
        save_model(model, f'best.pt')

    m.plot()

mlflow.log_artifact(f'{ckpt_path}/best.pt')

t = time.time() - t
print('Total', t, 'Avg', t / max_epoch)
