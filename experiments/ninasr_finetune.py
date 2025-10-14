import os
from typing import cast

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from models.fsrcnn.fsrcnn import FSRCNN_WITHOUT_PRETRAIN
from src.dataset import QuakeDataset
from src.trainer import Trainer

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    config = cast(DictConfig, OmegaConf.load("configs/config.yaml"))

    train_dataset = QuakeDataset(config, split="train")
    train_dataloader = DataLoader(train_dataset, **config.train_dataloader)

    val_dataset = QuakeDataset(config, split="val")
    val_dataloader = DataLoader(val_dataset, **config.val_dataloader)
    print(f"Scale: {config.training.scale}")
    model = FSRCNN_WITHOUT_PRETRAIN(scale=config.training.scale)

    trainer = Trainer(model, train_dataloader, val_dataloader, config)
    trainer.fit()
