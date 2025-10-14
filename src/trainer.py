import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.logger import logger
from src.metrics import Metrics, batch_metrics

# TODO: weight decay not applied
# TODO: learning rate scheduler not applied


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: DictConfig):
        self.device = config.training.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.training.optimizer.lr, weight_decay=config.training.optimizer.weight_decay)

        self.epochs = config.training.max_epochs

        # early stopping params
        self.patience = config.training.patience
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        lr, hr = batch
        lr, hr = lr.to(self.device), hr.to(self.device)

        sr = self.model(lr)
        loss = self.criterion(sr, hr)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        lr, hr = batch
        lr, hr = lr.to(self.device), hr.to(self.device)

        sr = self.model(lr)
        loss = self.criterion(sr, hr)
        return loss.item()

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_train_loss = 0.0
            epoch_train_metrics = Metrics()
            metric_steps = 0

            for i, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                epoch_train_loss += loss

                if i % 10 == 0:
                    logger.info(f"[Epoch {epoch}/{self.epochs}] Step {i}, Loss: {loss:.7f}")

                    sr_batch = self.model(batch[0].to(self.device))
                    metrics: Metrics = batch_metrics(sr_batch, batch[1].to(self.device))

                    epoch_train_metrics += metrics
                    metric_steps += 1

                    logger.info(f"Metrics - MSE: {metrics.mse:.4f}, PSNR: {metrics.psnr:.4f}, SSIM: {metrics.ssim:.4f}")

                    for param_group in self.optimizer.param_groups:
                        logger.info(f"Learning Rate: {param_group['lr']:.6f}")

            avg_loss = epoch_train_loss / len(self.train_loader)
            avg_metrics = epoch_train_metrics / metric_steps if metric_steps > 0 else Metrics()

            logger.info(f"Epoch {epoch} training phase finished. Avg Loss: {avg_loss:.7f}")
            logger.info(
                f"Train Metrics - MSE: {avg_metrics.mse:.4f}, PSNR: {avg_metrics.psnr:.4f}, SSIM: {avg_metrics.ssim:.4f}"
            )

            self.model.eval()
            epoch_val_loss = 0.0
            epoch_val_metrics = Metrics()
            val_metric_steps = 0

            with torch.no_grad():
                for j, batch in enumerate(self.val_loader):
                    loss = self.val_step(batch)
                    epoch_val_loss += loss

                    if j % 10 == 0:
                        logger.info(f"[Epoch {epoch}/{self.epochs}] Val Step {j}, Loss: {loss:.7f}")

                        sr_batch = self.model(batch[0].to(self.device))
                        metrics: Metrics = batch_metrics(sr_batch, batch[1].to(self.device))

                        epoch_val_metrics += metrics
                        val_metric_steps += 1

                        logger.info(
                            f"Val Metrics - MSE: {metrics.mse:.4f}, PSNR: {metrics.psnr:.4f}, SSIM: {metrics.ssim:.4f}"
                        )

            avg_val_loss = epoch_val_loss / len(self.val_loader)
            avg_val_metrics = epoch_val_metrics / val_metric_steps if val_metric_steps > 0 else Metrics()
            logger.info(f"Epoch {epoch} validation phase finished. Avg Val Loss: {avg_val_loss:.7f}")
            logger.info(
                f"Val Metrics - MSE: {avg_val_metrics.mse:.4f}, PSNR: {avg_val_metrics.psnr:.4f}, SSIM: {avg_val_metrics.ssim:.4f}"
            )

            # early stopping check
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_no_improve = 0
                torch.save(
                    self.model.state_dict(),
                    f"checkpoints/best_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
                )
                logger.info("Validation loss improved. Model saved.")
            else:
                self.epochs_no_improve += 1
                logger.info(f"No improvement in validation loss for {self.epochs_no_improve} epochs.")

                if self.epochs_no_improve >= self.patience:
                    logger.info(
                        f"Early stopping triggered after {epoch} epochs. Best val loss: {self.best_val_loss:.7f}"
                    )
                    break
