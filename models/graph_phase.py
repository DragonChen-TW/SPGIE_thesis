import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from tqdm import tqdm
import mlflow

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    mlflow_log: bool = True,
):
    model.train()
    scheduler.step()

    total_loss = 0.0
    total = 0
    correct = 0
    
    for data in tqdm(train_loader, total=len(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
    
        result = model(data)
        loss = F.nll_loss(result, data.y)
        loss.backward()

        if mlflow_log:
            mlflow.log_metric('train_loss', loss.item())
        
        # calculate
        total_loss += loss.item()
        pred = torch.argmax(result, dim=1)
        total += len(data.y)
        correct += (pred == data.y).sum().item()

        optimizer.step()
        scheduler.batch_step()
    acc = correct / total
    epoch_loss = total_loss / len(train_loader)
    return acc, epoch_loss

@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    mlflow_log: bool = True,
):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    
    for data in tqdm(test_loader, total=len(test_loader)):
        data = data.to(device)

        result = model(data)
        loss = F.nll_loss(result, data.y)
        if mlflow_log:
            mlflow.log_metric('test_loss', loss.item())
        
        # calculate
        total_loss += loss.item()
        pred = torch.argmax(result, dim=1)
        total += len(data.y)
        correct += (pred == data.y).sum().item()

    acc = correct / total
    epoch_loss = total_loss / len(test_loader)
    return acc, epoch_loss