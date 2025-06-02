import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from accelerate import Accelerator

import optuna
from FNO_torch.Diffusion.diffusion_basic import Diffusion

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
        loss = base_model.cal_loss(xb, yb)
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
            loss = base_model.cal_loss(xb, yb)
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


# --------------------------- Optuna Hyperparameter Tuning ---------------------------
def objective(model_cls, train_loader, test_loader, trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # 1) Sample hyperparameters
    config = {}
    config['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    config['batch_size'] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    config['timesteps'] = trial.suggest_int("timesteps", 50, 500, step=50)
    config['loss_ratio'] = trial.suggest_uniform("loss_ratio", 0.1, 0.9)
    config['weight_decay'] = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

    # 2) Build base network and diffusion model
    base_model = model_cls()
    diffusion_model = Diffusion(base_model, timesteps=config['timesteps'], loss_ratio=config['loss_ratio'])
    device = accelerator.device
    diffusion_model = diffusion_model.to(device)

    # 3) Build optimizer
    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 4) Use the provided data loaders
    valid_loader = test_loader

    # 5) Train for a fixed number of epochs, reporting validation loss to Optuna
    N_EPOCHS = 10
    best_val_loss = float("inf")
    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(diffusion_model, train_loader, optimizer, device)
        val_loss = valid_epoch(diffusion_model, valid_loader, device)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return best_val_loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Diffusion Model")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument("--storage", type=str, default=None,
                        help="SQLite URL to store Optuna study (e.g. 'sqlite:///optuna_study.db'), or None for in-memory")
    parser.add_argument("--prune", action="store_true", help="Enable Optuna pruning")
    args = parser.parse_args()

    # Create or load the study
    if args.storage is None:
        study = optuna.create_study(
            direction="minimize",
            study_name="diffusion_model_tuning",
            pruner=optuna.pruners.MedianPruner() if args.prune else optuna.pruners.NopPruner()
        )
    else:
        study = optuna.create_study(
            direction="minimize",
            study_name="diffusion_model_tuning",
            storage=args.storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner() if args.prune else optuna.pruners.NopPruner()
        )

    # Instantiate or load your actual data loaders and model class here:
    model_cls = YourNetwork  # pass the class, not an instance
    train_loader = ...       # replace with actual DataLoader
    test_loader = ...        # replace with actual DataLoader

    # Run the optimization
    study.optimize(lambda trial: objective(model_cls, train_loader, test_loader, trial), n_trials=args.n_trials)

    # Print the best trial
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Val Loss: {:.6f}".format(trial.value))
    print("  Best hyperparameters:")
    for key, val in trial.params.items():
        print(f"    {key}: {val}")

