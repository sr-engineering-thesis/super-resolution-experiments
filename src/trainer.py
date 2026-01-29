import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.logger import logger
from src.losses import get_loss_function
from src.metrics import Metrics, batch_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DictConfig,
        experiment_tracker=None,
        run_dir: str = "",
    ):
        self.device = config.training.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.run_dir = run_dir
        # self.initial_loss = config.training.loss_scales.get(config.training.loss, 1.0)
        self.initial_loss = None
        os.makedirs(self.run_dir, exist_ok=True)

        self.val_metrics = os.path.join(self.run_dir, "val_metrics.csv")
        self.train_metrics = os.path.join(self.run_dir, "train_metrics.csv")
        with open(self.val_metrics, "w") as f:
            f.write("epoch,loss,mse,psnr,ssim\n")

        with open(self.train_metrics, "w") as f:
            f.write("epoch,loss,mse,psnr,ssim\n")

        self.criterion: nn.Module = get_loss_function(config.training.loss, use_scaling=config.training.use_scaling).to(
            self.device
        )

        if config.training.optimizer.type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.training.optimizer.lr,
                weight_decay=config.training.optimizer.weight_decay,
            )
        elif config.training.optimizer.type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.training.optimizer.lr,
                momentum=0.9,
                weight_decay=config.training.optimizer.weight_decay,
            )
        else:
            logger.warning("Unknown optimizer type specified. Defaulting to AdamW.")
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.training.optimizer.lr,
                weight_decay=config.training.optimizer.weight_decay,
            )

        if config.training.scheduler.type == "cosine_annealing_warm_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.training.scheduler.T_0,
                T_mult=config.training.scheduler.T_mult,
                eta_min=config.training.scheduler.eta_min,
            )
        else:
            logger.warning("No scheduler or unknown scheduler type specified. Continuing without scheduler.")
            self.scheduler = None

        self.epochs = config.training.max_epochs
        self.patience = config.training.patience
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        lr, hr = batch
        lr, hr = lr.to(self.device), hr.to(self.device)

        sr = self.model(lr)
        loss_pixel = self.criterion(sr, hr)

        # if self.initial_loss is None:
        #     logger.info(f"Setting initial loss scale for {self.config.training.loss} loss: {self.config.training.loss_scales[self.config.training.loss]}")
        #     self.initial_loss = self.config.training.loss_scales[self.config.training.loss]

        # loss_pixel = loss_pixel / self.initial_loss

        self.optimizer.zero_grad()
        loss_pixel.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss_pixel.item()

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
                    sr_batch = self.model(batch[0].to(self.device))
                    metrics = batch_metrics(sr_batch, batch[1].to(self.device))
                    epoch_train_metrics += metrics
                    metric_steps += 1

                    logger.info(
                        f"[Epoch {epoch}] Step {i} Train Metrics - MSE: {metrics.mse:.4f}, PSNR: {metrics.psnr:.4f}, SSIM: {metrics.ssim:.4f}"
                    )

                    if self.experiment_tracker:
                        self.experiment_tracker.track(metrics.mse, name="train/mse", epoch=epoch)
                        self.experiment_tracker.track(metrics.psnr, name="train/psnr", epoch=epoch)
                        self.experiment_tracker.track(metrics.ssim, name="train/ssim", epoch=epoch)

            avg_train_loss = epoch_train_loss / len(self.train_loader)
            avg_train_metrics = epoch_train_metrics / metric_steps if metric_steps > 0 else Metrics()
            logger.info(
                f"[Epoch {epoch}] Avg Train Loss: {avg_train_loss:.6f} | MSE: {avg_train_metrics.mse:.4f}, "
                f"PSNR: {avg_train_metrics.psnr:.4f}, SSIM: {avg_train_metrics.ssim:.4f}"
            )

            if self.experiment_tracker:
                self.experiment_tracker.track(avg_train_loss, name="train/avg_loss", epoch=epoch)
                self.experiment_tracker.track(avg_train_metrics.mse, name="train/avg_mse", epoch=epoch)
                self.experiment_tracker.track(avg_train_metrics.psnr, name="train/avg_psnr", epoch=epoch)
                self.experiment_tracker.track(avg_train_metrics.ssim, name="train/avg_ssim", epoch=epoch)

            with open(self.train_metrics, "a") as f:
                f.write(
                    f"{epoch},{avg_train_loss:.8f},{avg_train_metrics.mse:.4f},{avg_train_metrics.psnr:.4f},{avg_train_metrics.ssim:.4f}\n"
                )

            # Validation
            self.model.eval()
            epoch_val_loss = 0.0
            epoch_val_metrics = Metrics()
            val_metric_steps = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    val_loss = self.val_step(batch)
                    epoch_val_loss += val_loss

                    sr_batch = self.model(batch[0].to(self.device))
                    metrics = batch_metrics(sr_batch, batch[1].to(self.device))
                    epoch_val_metrics += metrics
                    val_metric_steps += 1

                    logger.info(
                        f"[Epoch {epoch}] Val Metrics - MSE: {metrics.mse:.4f}, PSNR: {metrics.psnr:.4f}, SSIM: {metrics.ssim:.4f}"
                    )

                    if self.experiment_tracker:
                        self.experiment_tracker.track(metrics.mse, name="val/mse", epoch=epoch)
                        self.experiment_tracker.track(metrics.psnr, name="val/psnr", epoch=epoch)
                        self.experiment_tracker.track(metrics.ssim, name="val/ssim", epoch=epoch)

            avg_val_loss = epoch_val_loss / len(self.val_loader)
            avg_val_metrics = epoch_val_metrics / val_metric_steps if val_metric_steps > 0 else Metrics()
            logger.info(
                f"[Epoch {epoch}] Avg Val Loss: {avg_val_loss:.6f} | MSE: {avg_val_metrics.mse:.4f}, "
                f"PSNR: {avg_val_metrics.psnr:.4f}, SSIM: {avg_val_metrics.ssim:.4f}"
            )

            with open(self.val_metrics, "a") as f:
                f.write(
                    f"{epoch},{avg_val_loss:.8f},{avg_val_metrics.mse:.4f},{avg_val_metrics.psnr:.4f},{avg_val_metrics.ssim:.4f}\n"
                )

            if self.experiment_tracker:
                self.experiment_tracker.track(avg_val_loss, name="val/avg_loss", epoch=epoch)
                self.experiment_tracker.track(avg_val_metrics.mse, name="val/avg_mse", epoch=epoch)
                self.experiment_tracker.track(avg_val_metrics.psnr, name="val/avg_psnr", epoch=epoch)
                self.experiment_tracker.track(avg_val_metrics.ssim, name="val/avg_ssim", epoch=epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group["lr"]
                logger.info(f"Epoch {epoch} - Optimizer group {i} LR: {lr:.6f}")
                if self.experiment_tracker is not None:
                    self.experiment_tracker.track(lr, name=f"optimizer/lr_group_{i}", epoch=epoch)

            # Early stopping
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_no_improve = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.run_dir, f"checkpoints/best_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                    ),
                )
                logger.info(f"Validation improved. Model saved. Best Val Loss: {self.best_val_loss:.6f}")
            else:
                self.epochs_no_improve += 1
                logger.info(f"No improvement for {self.epochs_no_improve} epochs.")
                if self.epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
        return self.best_val_loss
