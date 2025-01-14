import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import mlflow
# 
from cnn_dataset import cnndataset_mapping
from models import lenet, resnet
from models.cnn_phase import train, test
from utils.meter import Meter
from utils.training import print_model_summary
from utils.scheduler import CyclicLRWithRestarts

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lenet')
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--batch_size', type=int, default=128)

# parser.add_argument('--hidden_dim', type=int, default=20)

parser.add_argument('--mlflow_log', type=bool, default=True)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--max_epoch', type=int, default=50)
args = parser.parse_args()
param = vars(args)

dataset_name = args.dataset_name
batch_size = args.batch_size

ckpt_path = f'ckpts/cnnckpt_{dataset_name}'
os.makedirs(ckpt_path, exist_ok=True)

print('dataset', dataset_name)

dataset_cls = cnndataset_mapping(dataset_name)

path = f'~/data/'

# TODO: fix following code

transform_train = T.Compose([
#     T.RandomAffine(0, translate=(0.1, 0.1)),
    T.RandomRotation(degrees=(-45, 45)),
#     T.ColorJitter(brightness=0.1, hue=0.2),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
])
transform_test = T.Compose([
    T.CenterCrop(32, padding=4),
    T.ToTensor(),
])

train_dataset = dataset_cls(path, train=True, transform=transform_train)
test_dataset = dataset_cls(path, train=False, transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

epoch_size = len(train_dataset)
# hidden_dim = args.hidden_dim

if dataset_name in ['mnist_m']:
    num_features = train_dataset.num_features
    output_dim = test_dataset.num_classes
else:
    if dataset_name in ['mnist', 'fashion_mnist']:
        num_features = 1
    else:
        num_features = 3
    output_dim = 10

print('Epoch size', epoch_size, 'Batch size', batch_size)
print('features ->', num_features)
# print('hidden_dim ->', hidden_dim)
print('classes ->', output_dim)


def save_model(model, ckpt):
    torch.save(model.state_dict(), f'{ckpt_path}/{ckpt}')

def load_model(model, ckpt):
    model.load_state_dict(torch.load(f'{ckpt_path}/{ckpt}', map_location=device))

model_name = args.model_name
if model_name == 'lenet':
    model = lenet.LeNet(num_features)
elif model_name == 'resnet18':
    model = resnet.resnet18(num_features)
elif model_name == 'resnet':
    model = resnet.resnet50(num_features)
elif model_name == 'resnext':
    model = resnet.resnext50_32x4d(num_features)


lr = float(args.lr)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.2, policy="cosine")

print_model_summary(model)

max_epoch = args.max_epoch
mlflow_log = args.mlflow_log
resume_training = False
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
        optimizer, criterion,        
        scheduler,
        device,
        mlflow_log=mlflow_log,
    )
    test_acc, test_loss = test(
        model, test_loader,
        criterion,
        device,
        mlflow_log=mlflow_log,
    )
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
        save_model(model, f'{model_name}_e{epoch:02}.pt')
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
