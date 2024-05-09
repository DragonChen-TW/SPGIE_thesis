import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import mlflow

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    mlflow_log: bool = True,
):
    model.train()
    if scheduler is not None:
        scheduler.step()

    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in tqdm(train_loader, total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        results = model(images)

        loss = criterion(results, labels)
        loss.backward()

        if mlflow_log:
            mlflow.log_metric('train_loss', loss.item())

        total_loss += loss.item()
        pred = torch.argmax(results, dim=1)
        total += images.shape[0]
        correct += (pred == labels).sum().item()

        optimizer.step()

    # Compute accuracy and loss of this epoch
    acc = correct / total
    epoch_loss = total_loss / len(train_loader)
    return acc, epoch_loss

@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    device: torch.device,
    mlflow_log: bool = True,
):
    model.eval()
    
    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device)
        labels = labels.to(device)

        results = model(images)
        loss = criterion(results, labels)
        if mlflow_log:
            mlflow.log_metric('test_loss', loss.item())

        total_loss += loss.item()
        pred = torch.argmax(results, dim=1)
        total += images.shape[0]
        correct += (pred == labels).sum().item()

    acc = correct / total
    epoch_loss = total_loss / len(test_loader)
    return acc, epoch_loss