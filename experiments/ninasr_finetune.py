import os
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from models.fsrcnn.fsrcnn import FSRCNN_BIG_PRETRAIN
from src.dataset import QuakeDataset
from src.logger import setup_aim_logger
from src.trainer import Trainer

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    config = cast(DictConfig, OmegaConf.load("configs/config.yaml"))

    experiment_tracker = setup_aim_logger("fsrccn_B_finetune_4x_experiment", config)

    train_dataset = QuakeDataset(config, split="train")
    train_dataloader = DataLoader(train_dataset, **config.train_dataloader)

    val_dataset = QuakeDataset(config, split="val")
    val_dataloader = DataLoader(val_dataset, **config.val_dataloader)
    # model = ninasr_b0(scale=4, pretrained=True)
    model = FSRCNN_BIG_PRETRAIN(scale=4)
    model.load_state_dict(torch.load("checkpoints/FSRCNN-x4.pt"))

    trainer = Trainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        config=config,
        experiment_tracker=experiment_tracker,
    )
    trainer.fit()
