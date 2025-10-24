import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.logger import logger
from src.metrics import Metrics, batch_metrics


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=("relu3_3",), use_input_norm=True):
        super().__init__()
        vgg_pretrained = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.layers = layers
        self.use_input_norm = use_input_norm

        if use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        self.vgg_layers = nn.ModuleDict()
        layer_map = {
            "relu1_1": 1, "relu1_2": 3, "relu2_1": 6, "relu2_2": 8,
            "relu3_1": 11, "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
            "relu4_1": 20, "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
            "relu5_1": 29, "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
        }

        for name in layers:
            self.vgg_layers[name] = nn.Sequential(*[vgg_pretrained[x] for x in range(layer_map[name] + 1)])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        features = {name: self.vgg_layers[name](x) for name in self.layers}
        return features


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DictConfig,
        experiment_tracker=None,  # Aim run
    ):
        self.device = config.training.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_tracker = experiment_tracker

        self.criterion = CharbonnierLoss()
        self.vgg_criterion = nn.L1Loss()
        self.vgg_extractor = VGGFeatureExtractor(layers=("relu3_3",)).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )

        self.epochs = config.training.max_epochs
        self.patience = config.training.patience
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        lr, hr = batch
        lr, hr = lr.to(self.device), hr.to(self.device)

        sr = self.model(lr)
        loss_pixel = self.criterion(sr, hr)

        sr_features = self.vgg_extractor(sr)
        hr_features = self.vgg_extractor(hr)
        loss_vgg = sum(self.vgg_criterion(sr_features[k], hr_features[k]) for k in sr_features)

        total_loss = loss_pixel + 0.21 * loss_vgg
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Logging
        logger.info(f"Step loss: {total_loss.item():.6f}")
        if self.experiment_tracker:
            self.experiment_tracker.track(total_loss.item(), name="train/loss", epoch=self.epochs_no_improve)

        return total_loss.item()

    def val_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        lr, hr = batch
        lr, hr = lr.to(self.device), hr.to(self.device)

        sr = self.model(lr)
        loss = self.criterion(sr, hr)

        logger.info(f"Validation step loss: {loss.item():.6f}")
        if self.experiment_tracker:
            self.experiment_tracker.track(loss.item(), name="val/loss", epoch=self.epochs_no_improve)

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

            if self.experiment_tracker:
                self.experiment_tracker.track(avg_val_loss, name="val/avg_loss", epoch=epoch)
                self.experiment_tracker.track(avg_val_metrics.mse, name="val/avg_mse", epoch=epoch)
                self.experiment_tracker.track(avg_val_metrics.psnr, name="val/avg_psnr", epoch=epoch)
                self.experiment_tracker.track(avg_val_metrics.ssim, name="val/avg_ssim", epoch=epoch)

            # Scheduler step
            self.scheduler.step()
            # inside the epoch loop, after scheduler.step()
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group['lr']
                logger.info(f"Epoch {epoch} - Optimizer group {i} LR: {lr:.6f}")
                if self.experiment_tracker is not None:
                    self.experiment_tracker.track(lr, name=f"optimizer/lr_group_{i}", epoch=epoch)

            # Early stopping
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_no_improve = 0
                torch.save(
                    self.model.state_dict(),
                    f"checkpoints/best_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
                )
                logger.info(f"Validation improved. Model saved. Best Val Loss: {self.best_val_loss:.6f}")
            else:
                self.epochs_no_improve += 1
                logger.info(f"No improvement for {self.epochs_no_improve} epochs.")
                if self.epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
