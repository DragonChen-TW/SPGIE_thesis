import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from tqdm import tqdm

def train(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
):
    model.train()
    scheduler.step()
    total_loss = 0.0
    correct = 0
    
    for data in tqdm(train_loader, total=len(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
    
        result = model(data)
        loss = F.nll_loss(result, data.y)
        loss.backward()
        
        # calculate
        total_loss += loss.item()
        pred = result.argmax(1)
        correct += pred.eq(data.y).sum().item()

        optimizer.step()
        scheduler.batch_step()
    return correct / len(train_dataset), total_loss / len(train_loader)

@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    
    for data in tqdm(test_loader, total=len(test_loader)):
        data = data.to(device)

        result = model(data)
        loss = F.nll_loss(result, data.y)
        
        # calculate
        total_loss += loss.item()
        pred = result.argmax(1)
        correct += pred.eq(data.y).sum().item()
        total += len(data.y)
    return correct / total, total_loss / len(test_loader)