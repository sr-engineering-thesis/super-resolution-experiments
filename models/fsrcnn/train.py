import hashlib
import logging
import time
from dataclasses import dataclass

import aim
import torch
from torch import nn
from torch.utils.data import DataLoader

from constants import CHECKPOINTS_DIR_PATH
from log_images import log_images
from metrics import im_mse, im_psnr, im_ssim
from models.fsrcnn.dataloader import QuakeDataset
from models.fsrcnn.fsrcnn import FSRCNN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using: %s", device)

if device.type == "cuda":
    logger.info(torch.cuda.get_device_name(0))
    logger.info("Memory Usage:")
    logger.info("Allocated: %s GB", round(torch.cuda.memory_allocated(0) / 1024**3, 1))
    logger.info("Cached:   %s GB", round(torch.cuda.memory_reserved(0) / 1024**3, 1))

hash_for_run = hashlib.md5(str(time.time()).encode()).hexdigest()


def save_model_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float
):
    path = f"{CHECKPOINTS_DIR_PATH}/{hash_for_run}_{model.__class__.__name__}_epoch_{epoch + 1}.pth"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    logger.info(f"Model checkpoint saved to {path}")


@dataclass
class FSRCNNConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 30
    loss_fn: nn.Module = nn.MSELoss()
    upscale_factor: int = 2
    optimizer: str = "adam"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    color_space: str = "YCrCb"

    def log_to_aim(self):
        for field_name in self.__dataclass_fields__.keys():
            run_aim[field_name] = getattr(self, field_name)


METRICS = {
    "mse": im_mse,
    "psnr": im_psnr,
    "ssim": im_ssim,
}

run_aim = aim.Run(
    experiment="FSRCNN - full dataset train test",
    repo="aim://server.aim.mjanicki.org:53800",
)
FSRCNNConfig().log_to_aim()


def get_loaders(config):
    train_dataset = QuakeDataset(
        device=config.device, split="train", load_to_ram=True, augment=True
    )
    val_dataset = QuakeDataset(device=config.device, split="val", load_to_ram=True)
    test_dataset = QuakeDataset(device=config.device, split="test", load_to_ram=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


# --- Training Step ---
def train_one_epoch(model, loader, optimizer, criterion, config, metrics):
    model.train()
    total_loss = 0.0
    metric_sums = {k: 0.0 for k in metrics}
    for i, (lr, hr) in enumerate(loader):
        lr, hr = lr.to(config.device), hr.to(config.device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        for k, func in metrics.items():
            metric_sums[k] += func(sr.detach().cpu().numpy(), hr.detach().cpu().numpy())
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metric_sums.items()}


# --- Validation/Test Step ---
@torch.no_grad()
def evaluate(model, loader, criterion, config, metrics):
    model.eval()
    total_loss = 0.0
    metric_sums = {k: 0.0 for k in metrics}
    total_eval_time_ns = 0

    for lr, hr in loader:
        lr, hr = lr.to(config.device), hr.to(config.device)

        start_ns = time.monotonic_ns()
        sr = model(lr)
        elapsed_ns = time.monotonic_ns() - start_ns
        total_eval_time_ns += elapsed_ns

        loss = criterion(sr, hr)
        total_loss += loss.item()

        sr_np = sr.detach().cpu().numpy()
        hr_np = hr.detach().cpu().numpy()
        for k, func in metrics.items():
            metric_sums[k] += func(sr_np, hr_np)

    n = len(loader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in metric_sums.items()}
    avg_eval_time_ms = total_eval_time_ns / n / 1e6

    logger.info(
        f"Average evaluation time per image: {avg_eval_time_ms:.2f} ms"
    )  # batch size for test dataloader is 1
    run_aim.track(
        avg_eval_time_ms, name="eval_time", epoch=0, context={"subset": "test"}
    )

    return avg_loss, avg_metrics


# --- Training Loop ---
def train_model():
    config = FSRCNNConfig()
    logger.info(f"Using device: {config.device}")
    model = FSRCNN(config.upscale_factor).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = config.loss_fn.to(config.device)
    train_loader, val_loader, test_loader = get_loaders(config)

    for epoch in range(config.num_epochs):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, config, METRICS
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, config, METRICS)

        logger.info(f"Epoch [{epoch + 1}/{config.num_epochs}]")
        logger.info(
            f"  Train Loss: {train_loss:.4f} | "
            + " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
        )
        logger.info(
            f"  Val   Loss: {val_loss:.4f} | "
            + " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
        )

        run_aim.track(
            train_loss, name="loss", epoch=epoch + 1, context={"subset": "train"}
        )
        run_aim.track(val_loss, name="loss", epoch=epoch + 1, context={"subset": "val"})

        for k in METRICS.keys():
            run_aim.track(
                train_metrics[k], name=k, epoch=epoch + 1, context={"subset": "train"}
            )
            run_aim.track(
                val_metrics[k], name=k, epoch=epoch + 1, context={"subset": "val"}
            )

        if (epoch + 1) % 5 == 0:
            save_model_checkpoint(model, optimizer, epoch, train_loss)
            log_images(model, epoch + 1, config, hash_for_run)

    test_loss, test_metrics = evaluate(model, test_loader, criterion, config, METRICS)
    logger.info("\nTest Results:")
    logger.info(
        f"  Test Loss: {test_loss:.4f} | "
        + " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())
    )

    run_aim.track(
        test_loss, name="loss", epoch=config.num_epochs, context={"subset": "test"}
    )
    for k, v in test_metrics.items():
        run_aim.track(v, name=k, epoch=config.num_epochs, context={"subset": "test"})


if __name__ == "__main__":
    train_model()
