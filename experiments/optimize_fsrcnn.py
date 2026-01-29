import optuna
from hydra import compose, initialize

from experiments.fsrcnn_finetune import train_fsrcnn


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("training.optimizer.lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("training.optimizer.weight_decay", 0.0, 1e-3)
    patch_size = trial.suggest_categorical("data.patch_size", [32, 64, 96, 128, 192])

    T_0 = trial.suggest_int("training.scheduler.T_0", 10, 50)
    T_mult = trial.suggest_int("training.scheduler.T_mult", 1, 4)
    eta_min = trial.suggest_float("training.scheduler.eta_min", 1e-7, 1e-5, log=True)

    overrides = [
        f"training.optimizer.lr={lr}",
        f"training.optimizer.weight_decay={weight_decay}",
        f"data.patch_size={patch_size}",
        f"training.scheduler.T_0={T_0}",
        f"training.scheduler.T_mult={T_mult}",
        f"training.scheduler.eta_min={eta_min}",
    ]

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    cfg.output_dir = f"optuna_runs/trial_{trial.number}"
    val_loss = train_fsrcnn(cfg)

    trial.set_user_attr("val_loss", val_loss)
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)