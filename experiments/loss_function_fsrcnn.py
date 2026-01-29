import os

import torch
from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models.fsrcnn.fsrcnn import FSRCNN_BIG_PRETRAIN
from src.dataset import QuakeDataset
from src.logger import setup_aim_logger
from src.trainer import Trainer


@main(config_path="../configs", config_name="config", version_base="1.3")
def train_fsrcnn(config: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    experiment_tracker = setup_aim_logger("loss_functions_test", config)

    train_dataset = QuakeDataset(config, split="train")
    train_dataloader = DataLoader(train_dataset, **config.train_dataloader)

    val_dataset = QuakeDataset(config, split="val")
    val_dataloader = DataLoader(val_dataset, **config.val_dataloader)

    model = FSRCNN_BIG_PRETRAIN(scale=config.training.scale)
    model.load_state_dict(torch.load("checkpoints/FSRCNN-x2.pt"))

    trainer = Trainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        config=config,
        experiment_tracker=experiment_tracker,
        run_dir=output_dir,
    )

    best_val_loss = trainer.fit()
    return best_val_loss

if __name__ == "__main__":
    train_fsrcnn()
