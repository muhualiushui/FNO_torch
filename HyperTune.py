import torch
import torch.nn as nn
from accelerate import Accelerator
from FNO_torch.train import train_epoch, valid_epoch
import optuna
from FNO_torch.Diffusion.diffusion_basic import Diffusion
from tqdm.auto import tqdm

accelerator = Accelerator()

def objective(model_cls, train_loader, test_loader, x_name, y_name, N_EPOCHS, trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter tuning.
    """
    config = {}
    config['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    config['batch_size'] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    config['timesteps'] = trial.suggest_int("timesteps", 300, 700, step=50)
    config['weight_decay'] = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

    base_model = model_cls
    diffusion_model = nn.DataParallel(Diffusion(base_model, timesteps=config['timesteps'], loss_ratio=0.8))
    device = accelerator.device
    diffusion_model = diffusion_model.to(device)

    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    valid_loader = test_loader

    best_val_loss = float("inf")
    for epoch in tqdm(range(N_EPOCHS), desc='Epoch', unit='epoch', leave=True, dynamic_ncols=True, position=0):
        train_loss = train_epoch(diffusion_model, train_loader, optimizer, device, x_name=x_name, y_name=y_name)
        val_loss = valid_epoch(diffusion_model, valid_loader, device, x_name=x_name, y_name=y_name)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return best_val_loss
