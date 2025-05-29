import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from accelerate import Accelerator
accelerator = Accelerator()

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: Optimizer,
                device: torch.device,
                *,
                x_name: str | None = None,
                y_name: str | None = None) -> float:
    model.train()
    running = 0.0
    total = 0
    pbar = tqdm(train_loader, desc='Train', position=1, leave=False, dynamic_ncols=True)
    for batch in pbar:
        if isinstance(batch, dict):
            if x_name is None or y_name is None:
                raise ValueError("When batches are dictionaries, x_name and y_name must be provided.")
            xb = batch[x_name].to(device)
            yb = batch[y_name].to(device)
        elif isinstance(batch, (list, tuple)):
            xb = batch[0].to(device)
            yb = batch[1].to(device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        optimizer.zero_grad()
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        loss = base_model.Diffusion(yb, xb)
        accelerator.backward(loss)
        optimizer.step()
        running += loss.item() * xb.size(0)
        total += xb.size(0)
    return running / total

def valid_epoch(model: nn.Module,
                test_loader: DataLoader,
                device: torch.device,
                *,
                x_name: str | None = None,
                y_name: str | None = None) -> float:
    model.eval()
    val_running = 0.0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Valid', position=1, leave=False, dynamic_ncols=True)
        for batch in pbar:
            if isinstance(batch, dict):
                if x_name is None or y_name is None:
                    raise ValueError("When batches are dictionaries, x_name and y_name must be provided.")
                xb = batch[x_name].to(device)
                yb = batch[y_name].to(device)
            elif isinstance(batch, (list, tuple)):
                xb = batch[0].to(device)
                yb = batch[1].to(device)
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            loss = base_model.Diffusion(yb, xb)
            val_running += loss.item() * xb.size(0)
            total += xb.size(0)
    return val_running / total

def train_model(model: nn.Module,
                train_loader: DataLoader,
                test_loader: DataLoader,
                optimizer: Optimizer,
                epochs: int,
                device: torch.device,
                x_name: str = None,
                y_name: str = None) -> dict:
    # Prepare model, optimizer, and data loaders for multi-GPU
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    device = accelerator.device
    history = {'train_loss': [], 'val_loss': []}
    pbar = tqdm(range(epochs), desc='Epoch', unit='epoch', leave=True, dynamic_ncols=True, position=0)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, device, x_name=x_name, y_name=y_name)
        val_loss = valid_epoch(model, test_loader, device, x_name=x_name, y_name=y_name)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
    return history