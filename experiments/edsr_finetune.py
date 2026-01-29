import os
from typing import cast

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchsr.models import edsr_r16f64 as EDSR

from src.dataset import QuakeDataset
from src.trainer import Trainer

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    config = cast(DictConfig, OmegaConf.load("configs/config.yaml"))

    train_dataset = QuakeDataset(config, split="train")
    train_dataloader = DataLoader(train_dataset, **config.train_dataloader)

    val_dataset = QuakeDataset(config, split="val")
    val_dataloader = DataLoader(val_dataset, **config.val_dataloader)

    model = EDSR(scale=config.training.scale, pretrained=True)

    trainer = Trainer(model, train_dataloader, val_dataloader, config)
    trainer.fit()
