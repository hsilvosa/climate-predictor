"""PyTorch CUDA training loop, early stopping, and model checkpointing."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from climate_forecast.config import CHECKPOINTS_DIR, ModelConfig, TrainingConfig
from climate_forecast.data.dataset import ClimateDataScaler
from climate_forecast.models.loss import ClimateCompositeLoss
from climate_forecast.models.spatiotemporal_net import ClimateSpatiotemporalNet
from climate_forecast.training.metrics import ClimateEvaluationMetrics, evaluate_predictions

logger = logging.getLogger(__name__)


class ClimateForecasterTrainer:
    """Trains the ClimateSpatiotemporalNet model with early stopping and CUDA acceleration."""

    def __init__(
        self,
        model: ClimateSpatiotemporalNet,
        training_cfg: Optional[TrainingConfig] = None,
        model_cfg: Optional[ModelConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.training_cfg = training_cfg or TrainingConfig()
        self.model_cfg = model_cfg or model.config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.criterion = ClimateCompositeLoss(self.training_cfg).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_cfg.learning_rate,
            weight_decay=self.training_cfg.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.training_cfg.scheduler_factor,
            patience=self.training_cfg.scheduler_patience,
            min_lr=1e-6,
        )

        self.history: Dict[str, list] = {
            "train_total_loss": [],
            "train_pinball_loss": [],
            "train_hazard_loss": [],
            "val_total_loss": [],
            "val_pinball_loss": [],
            "val_hazard_loss": [],
            "learning_rate": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss_sum = 0.0
        pinball_loss_sum = 0.0
        hazard_loss_sum = 0.0
        num_batches = len(train_loader)

        for batch in train_loader:
            x_seq = batch["x_seq"].to(self.device)
            x_spatial = batch["x_spatial"].to(self.device)
            x_future_time = batch["x_future_time"].to(self.device)
            y_targets = batch["y_targets"].to(self.device)
            y_hazards = batch["y_hazards"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(x_seq, x_spatial, x_future_time)
            loss_dict = self.criterion(
                outputs["quantiles"],
                outputs["hazard_probs"],
                y_targets,
                y_hazards,
            )

            loss = loss_dict["total_loss"]
            loss.backward()

            # Gradient clipping
            if self.training_cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training_cfg.grad_clip_norm)

            self.optimizer.step()

            total_loss_sum += loss.item()
            pinball_loss_sum += loss_dict["pinball_loss"].item()
            hazard_loss_sum += loss_dict["hazard_loss"].item()

        return {
            "total_loss": total_loss_sum / max(1, num_batches),
            "pinball_loss": pinball_loss_sum / max(1, num_batches),
            "hazard_loss": hazard_loss_sum / max(1, num_batches),
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[Dict[str, float], ClimateEvaluationMetrics]:
        self.model.eval()
        total_loss_sum = 0.0
        pinball_loss_sum = 0.0
        hazard_loss_sum = 0.0
        num_batches = len(data_loader)

        all_q_preds = []
        all_prob_preds = []
        all_y_targets = []
        all_y_hazards = []

        for batch in data_loader:
            x_seq = batch["x_seq"].to(self.device)
            x_spatial = batch["x_spatial"].to(self.device)
            x_future_time = batch["x_future_time"].to(self.device)
            y_targets = batch["y_targets"].to(self.device)
            y_hazards = batch["y_hazards"].to(self.device)

            outputs = self.model(x_seq, x_spatial, x_future_time)
            loss_dict = self.criterion(
                outputs["quantiles"],
                outputs["hazard_probs"],
                y_targets,
                y_hazards,
            )

            total_loss_sum += loss_dict["total_loss"].item()
            pinball_loss_sum += loss_dict["pinball_loss"].item()
            hazard_loss_sum += loss_dict["hazard_loss"].item()

            all_q_preds.append(outputs["quantiles"].cpu().numpy())
            all_prob_preds.append(outputs["hazard_probs"].cpu().numpy())
            all_y_targets.append(y_targets.cpu().numpy())
            all_y_hazards.append(y_hazards.cpu().numpy())

        losses = {
            "total_loss": total_loss_sum / max(1, num_batches),
            "pinball_loss": pinball_loss_sum / max(1, num_batches),
            "hazard_loss": hazard_loss_sum / max(1, num_batches),
        }

        q_preds_arr = np.concatenate(all_q_preds, axis=0) if all_q_preds else np.zeros((0, 14, 3, 3))
        prob_preds_arr = np.concatenate(all_prob_preds, axis=0) if all_prob_preds else np.zeros((0, 14, 3))
        y_tgt_arr = np.concatenate(all_y_targets, axis=0) if all_y_targets else np.zeros((0, 14, 3))
        y_haz_arr = np.concatenate(all_y_hazards, axis=0) if all_y_hazards else np.zeros((0, 14, 3))

        metrics = evaluate_predictions(q_preds_arr, prob_preds_arr, y_tgt_arr, y_haz_arr)
        return losses, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path | str = CHECKPOINTS_DIR,
        scaler: Optional[ClimateDataScaler] = None,
    ) -> Tuple[ClimateSpatiotemporalNet, Dict]:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        best_model_file = ckpt_path / "best_climate_forecaster.pt"

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting training on {self.device} for {self.training_cfg.num_epochs} epochs...")

        for epoch in range(1, self.training_cfg.num_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_losses, val_metrics = self.evaluate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.scheduler.step(val_losses["total_loss"])

            # Log history
            self.history["train_total_loss"].append(train_metrics["total_loss"])
            self.history["train_pinball_loss"].append(train_metrics["pinball_loss"])
            self.history["train_hazard_loss"].append(train_metrics["hazard_loss"])
            self.history["val_total_loss"].append(val_losses["total_loss"])
            self.history["val_pinball_loss"].append(val_losses["pinball_loss"])
            self.history["val_hazard_loss"].append(val_losses["hazard_loss"])
            self.history["learning_rate"].append(current_lr)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:02d}/{self.training_cfg.num_epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_metrics['total_loss']:.4f} - Val Loss: {val_losses['total_loss']:.4f} "
                f"(TMAX MAE: {val_metrics.tmax_metrics.mae_overall:.2f}C, HW F1: {val_metrics.heatwave_hazard.f1:.3f}) - "
                f"LR: {current_lr:.6f}"
            )

            # Checkpoint best model
            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                patience_counter = 0

                state = {
                    "model_state_dict": self.model.state_dict(),
                    "model_config": asdict(self.model_cfg),
                    "training_config": asdict(self.training_cfg),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "val_metrics": val_metrics.to_dict(),
                }
                if scaler:
                    state["scaler_params"] = asdict(scaler.params)

                torch.save(state, best_model_file)
                logger.info(f"Saved new best model checkpoint to {best_model_file}")
            else:
                patience_counter += 1
                if patience_counter >= self.training_cfg.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        # Reload best model weights
        if best_model_file.exists():
            checkpoint = torch.load(best_model_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        return self.model, self.history
