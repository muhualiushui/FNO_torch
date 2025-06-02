import torch
import torch.nn as nn
from accelerate import Accelerator
from FNO_torch.train import train_epoch, valid_epoch
import optuna
from FNO_torch.Diffusion.diffusion_basic import Diffusion
from tqdm.auto import tqdm

accelerator = Accelerator()

# --------------------------- Optuna Hyperparameter Tuning ---------------------------
def objective(model_cls, train_loader, test_loader, x_name, y_name, N_EPOCHS, trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # 1) Sample hyperparameters
    config = {}
    config['learning_rate'] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    config['batch_size'] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    config['timesteps'] = trial.suggest_int("timesteps", 300, 700, step=50)
    config['weight_decay'] = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

    # 2) Build base network and diffusion model
    base_model = model_cls
    diffusion_model = nn.DataParallel(Diffusion(base_model, timesteps=config['timesteps'], loss_ratio=0.8))
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


if __name__ == "__main__":
    # 1) Set your hyperparameter tuning settings directly here
    n_trials = 20
    use_prune = False  # Set to True to enable pruning (early stopping of unpromising trials)

    # 2) Create the Optuna study with or without pruning
    study = optuna.create_study(
        direction="minimize",
        study_name="diffusion_model_tuning",
        pruner=optuna.pruners.MedianPruner() if use_prune else optuna.pruners.NopPruner()
    )

    # 3) Instantiate or load your actual data loaders and model class here:
    model_cls = YourNetwork  # Replace with your actual network class
    train_loader = ...       # Replace with your actual DataLoader for training
    test_loader = ...        # Replace with your actual DataLoader for validation

    # 4) Run the optimization
    study.optimize(
        lambda trial: objective(model_cls, train_loader, test_loader, trial),
        n_trials=n_trials
    )

    # 5) Print the best trial results
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Val Loss: {trial.value:.6f}")
    print("  Best hyperparameters:")
    for key, val in trial.params.items():
        print(f"    {key}: {val}")