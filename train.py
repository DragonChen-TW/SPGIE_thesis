import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import mlflow
# 
from spdataset import spdataset_mapping
from models.jit_drn_model import DynamicReductionNetworkJit
from models.graph_phase import train, test
from utils.meter import Meter
from utils.training import print_model_summary
from utils.scheduler import CyclicLRWithRestarts

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--with_std', type=bool, default=False)
parser.add_argument('--num_sp', type=int, default=75)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--hidden_dim', type=int, default=20)
parser.add_argument('--drn_k', type=int, default=4)

parser.add_argument('--mlflow_log', type=bool, default=True)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--max_epoch', type=int, default=20)
args = parser.parse_args()
param = vars(args)

dataset_name = args.dataset_name
num_superpixel = int(args.num_sp)
batch_size = args.batch_size

with_std = args.with_std

param.update({
    'num_superpixel': num_superpixel,
    'with_std': with_std,
})

ckpt_path = f'ckpts/ckpt_{dataset_name}'
os.makedirs(ckpt_path, exist_ok=True)

print('dataset', dataset_name, num_superpixel)
if with_std:
    print('w/ std')

dataset_cls = spdataset_mapping(dataset_name)

path = f'~/data/{dataset_name.upper()}_SUPERPIXEL'
if with_std:
    path = f'~/data/{dataset_name.upper()}_SUPERPIXEL_WSTD'
if num_superpixel != 75:
    path = path.replace('_SUPERPIXEL', f'_{num_superpixel}_SUPERPIXEL')

# transform = T.Cartesian(cat=False)
transform = None
train_dataset = dataset_cls(path, train=True, transform=transform)
test_dataset = dataset_cls(path, train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

epoch_size = len(train_dataset)
num_features = train_dataset.num_features
if with_std:
    num_features = train_dataset.num_features * 2
hidden_dim = args.hidden_dim
output_dim = test_dataset.num_classes

print('Epoch', epoch_size, 'Batch size', batch_size)
print('features ->', num_features)
print('hidden_dim ->', hidden_dim)
print('classes ->', output_dim)

drn_k = int(args.drn_k)
aggr = 'add'
pool = 'max'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drn = DynamicReductionNetworkJit(
            input_dim=num_features, hidden_dim=hidden_dim,
            output_dim=output_dim,
            k=drn_k, aggr=aggr,
            pool=pool,
            agg_layers=2, mp_layers=2, in_layers=3, out_layers=3,
            graph_features=num_features,
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

def save_model(model, ckpt):
    torch.save(model.state_dict(), f'{ckpt_path}/{ckpt}')

def load_model(model, ckpt):
    model.load_state_dict(torch.load(f'{ckpt_path}/{ckpt}', map_location=device))

lr = float(args.lr)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.2, policy="cosine")

print_model_summary(model)

max_epoch = args.max_epoch
mlflow_log = args.mlflow_log
# resume_training = False
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
    train_acc, train_loss = train(
        model, train_loader,
        optimizer, scheduler, device,
        mlflow_log=mlflow_log,
    )
    test_acc, test_loss = test(model, test_loader, device, mlflow_log=mlflow_log)
    m.update(train_loss, train_acc, test_loss, test_acc)
    print('Epoch: {:02d}, Train: {:.4f} Test: {:.4f}'.format(epoch, train_acc, test_acc))

    if mlflow_log:
        mlflow.log_metrics({
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
        }, step=epoch - 1)

    if epoch % 5 == 0:
        save_model(model, f'e{epoch:02}.pt')
    if test_acc > best_acc:
        best_acc = test_acc
        print('best in epoch', epoch)
        print('test acc', test_acc)
        save_model(model, f'best.pt')

    m.plot()

if mlflow_log:
    mlflow.log_metric('best_acc', best_acc)
    mlflow.log_artifact(f'{ckpt_path}/best.pt')

t = time.time() - t
print('Total', t, 'Avg', t / max_epoch)
