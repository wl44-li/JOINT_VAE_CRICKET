"""Train a JointVAE on the genotypes of one fold. No label enters this file."""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from losses import (
    capacity,
    joint_vae_objective,
    kl_categorical,
    kl_categorical_marginal,
    kl_gaussian,
    reconstruction_loss,
)
from model import encode_genotype


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 400
    lr: float = 5e-4
    weight_decay: float = 0.0

    # Capacity objective. The capacity of z rises from 0 to capacity_z, the capacity of c from 0 to
    # log(K), both over the first ramp_fraction of the training steps.
    gamma_z: float = 1000.0
    capacity_z: float = 1.0
    gamma_c: Optional[float] = None  # None: same as gamma_z
    lambda_marginal: float = 0.0
    ramp_fraction: float = 0.5

    # "last": the model after the last epoch (the objective changes during the ramp, so the last
    # epoch is the model the schedule is designed to give).
    # "best_val": lowest validation loss among the epochs after the ramp.
    select: str = "last"

    verbose: bool = True
    log_every: int = 50


class JointVAETrainer:
    """Train a JointVAE on uint8 dosage arrays (3 = missing)."""

    def __init__(self, model, cfg: TrainConfig, device=None):
        self.cfg = cfg

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.gamma_c = cfg.gamma_z if cfg.gamma_c is None else cfg.gamma_c
        self.capacity_c_max = math.log(model.n_categories)

        optimizer_class = AdamW if cfg.weight_decay > 0 else Adam
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=self.device.type == "cuda",
        )

        self.history = []
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_state_dict = None
        self.reference = None

    def _tensor(self, dosage):
        return torch.as_tensor(np.asarray(dosage), dtype=torch.uint8, device=self.device)

    def _forward_losses(self, dosage, sample, capacity_z, capacity_c):
        cfg = self.cfg

        x, mask = encode_genotype(dosage)
        recon_logits, mu, logvar, category_logits = self.model(x, mask, sample=sample)

        loss_recon = reconstruction_loss(recon_logits, x, mask).mean()
        loss_kl_z = kl_gaussian(mu, logvar).sum(-1).mean()
        loss_kl_c = kl_categorical(category_logits).mean()
        loss_kl_marginal = kl_categorical_marginal(category_logits)

        loss = joint_vae_objective(
            loss_recon=loss_recon,
            loss_kl_z=loss_kl_z,
            loss_kl_c=loss_kl_c,
            loss_kl_marginal=loss_kl_marginal,
            capacity_z=capacity_z,
            capacity_c=capacity_c,
            gamma_z=cfg.gamma_z,
            gamma_c=self.gamma_c,
            lambda_marginal=cfg.lambda_marginal,
        )

        metrics = {
            "loss": loss.detach(),
            "recon": loss_recon.detach(),
            "kl_z": loss_kl_z.detach(),
            "kl_c": loss_kl_c.detach(),
            "kl_marginal": loss_kl_marginal.detach(),
        }

        return loss, metrics

    @torch.no_grad()
    def _final_objective(self, dosage):
        """Objective with the final capacities, z = mean and c = arg-max."""
        self.model.eval()

        _, metrics = self._forward_losses(
            dosage,
            sample=False,
            capacity_z=self.cfg.capacity_z,
            capacity_c=self.capacity_c_max,
        )

        return {name: value.item() for name, value in metrics.items()}

    def fit(self, x_train, x_val=None):
        """Train on x_train. x_val is only used to follow (and optionally select by) its loss."""
        cfg = self.cfg

        if cfg.select not in ("last", "best_val"):
            raise ValueError("select must be 'last' or 'best_val'.")

        train = self._tensor(x_train)
        val = self._tensor(x_val) if x_val is not None and len(x_val) > 0 else None

        if cfg.select == "best_val" and val is None:
            raise ValueError("select='best_val' needs a validation set.")

        # Allele-frequency information of the training crickets only.
        x, mask = encode_genotype(train)
        feature_mean = x.sum(0) / mask.sum(0).clamp_min(1)
        del x, mask

        self.model.set_data_stats(feature_mean)
        self.reference = feature_mean.clamp(1e-4, 1 - 1e-4)

        n_train = train.shape[0]
        steps_per_epoch = math.ceil(n_train / cfg.batch_size)
        ramp_steps = int(cfg.ramp_fraction * cfg.max_epochs * steps_per_epoch)

        self.history = []
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_state_dict = None

        step = 0

        for epoch in range(1, cfg.max_epochs + 1):
            self.model.train()

            order = torch.randperm(n_train, device=self.device)
            totals = {}

            for batch_id in range(steps_per_epoch):
                batch = order[batch_id * cfg.batch_size:(batch_id + 1) * cfg.batch_size]

                loss, metrics = self._forward_losses(
                    train[batch],
                    sample=True,
                    capacity_z=capacity(step, ramp_steps, cfg.capacity_z),
                    capacity_c=capacity(step, ramp_steps, self.capacity_c_max),
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                step += 1

                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + value

            # One transfer from the GPU per epoch, not per step.
            totals = {name: value.item() for name, value in totals.items()}

            if not math.isfinite(totals["loss"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")

            row = {
                "epoch": epoch,
                "capacity_z": capacity(step, ramp_steps, cfg.capacity_z),
                "capacity_c": capacity(step, ramp_steps, self.capacity_c_max),
            }

            row.update({
                f"train_{name}": value / steps_per_epoch
                for name, value in totals.items()
            })

            if val is not None:
                val_metrics = self._final_objective(val)

                row.update({
                    f"val_{name}": value
                    for name, value in val_metrics.items()
                })

                if (
                    cfg.select == "best_val"
                    and step >= ramp_steps
                    and val_metrics["loss"] < self.best_val_loss
                ):
                    self.best_val_loss = val_metrics["loss"]
                    self.best_epoch = epoch

                    self.best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }

            self.history.append(row)

            if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == cfg.max_epochs):
                print(
                    f"Epoch {epoch:03d}/{cfg.max_epochs} "
                    f"train_loss={row['train_loss']:.2f} "
                    f"train_recon={row['train_recon']:.2f} "
                    f"kl_z={row['train_kl_z']:.3f} "
                    f"kl_c={row['train_kl_c']:.3f} "
                    f"val_loss={row.get('val_loss', float('nan')):.2f}"
                )

        if cfg.select == "last":
            self.best_epoch = cfg.max_epochs
            self.best_val_loss = self.history[-1].get("val_loss", float("nan"))

        return self.history

    def restore_best_model(self):
        """Load the selected weights (nothing to do when the last epoch is used)."""
        if self.cfg.select == "last":
            return

        if self.best_state_dict is None:
            raise RuntimeError("No best model is available. Run fit() first.")

        self.model.load_state_dict(self.best_state_dict)

    @torch.no_grad()
    def evaluate(self, x, batch_size=128):
        """Label-free numbers of one split, with z = mean and c = arg-max."""
        dosage = self._tensor(x)

        if dosage.shape[0] == 0:
            return {}

        self.model.eval()

        recon, n_observed, reference, kl_z, probabilities = [], [], [], [], []

        for start in range(0, dosage.shape[0], batch_size):
            x_batch, mask = encode_genotype(dosage[start:start + batch_size])
            recon_logits, mu, logvar, category_logits = self.model(x_batch, mask, sample=False)

            recon.append(reconstruction_loss(recon_logits, x_batch, mask))
            n_observed.append(mask.sum(-1))
            reference.append(
                (
                    F.binary_cross_entropy(
                        self.reference.expand_as(x_batch), x_batch, reduction="none"
                    ) * mask
                ).sum(-1)
            )
            kl_z.append(kl_gaussian(mu, logvar))
            probabilities.append(F.softmax(category_logits, dim=-1))

        recon = torch.cat(recon)
        n_observed = torch.cat(n_observed).clamp_min(1)
        reference = torch.cat(reference)
        kl_z = torch.cat(kl_z)
        q = torch.cat(probabilities).double().clamp_min(1e-12)

        kl_c = (q * (q.log() + math.log(q.shape[1]))).sum(-1)
        mutual_information = (q * (q.log() - q.mean(0, keepdim=True).log())).sum(-1)

        metrics = self._final_objective(dosage)

        metrics.update({
            # Reconstruction per observed input entry, and the same for a model that
            # only knows the training allele frequencies.
            "recon_per_feature": (recon / n_observed).mean().item(),
            "allele_frequency_recon_per_feature": (reference / n_observed).mean().item(),
            "n_active_z": int((kl_z.mean(0) > 0.01).sum().item()),
            "kl_c": kl_c.mean().item(),
            "category_information": mutual_information.mean().item(),
            "categories_used": int(q.argmax(-1).unique().numel()),
        })

        return metrics

    @torch.no_grad()
    def encode_dataset(self, x, batch_size=128):
        """Posterior mean and log-variance of z and q(c|x), in the stored sample order."""
        dosage = self._tensor(x)

        was_training = self.model.training
        self.model.eval()

        outputs = {"mu": [], "logvar": [], "probs": []}

        try:
            for start in range(0, dosage.shape[0], batch_size):
                x_batch, mask = encode_genotype(dosage[start:start + batch_size])
                mu, logvar, category_logits = self.model.encode(x_batch, mask)

                outputs["mu"].append(mu.cpu().numpy())
                outputs["logvar"].append(logvar.cpu().numpy())
                outputs["probs"].append(F.softmax(category_logits, dim=-1).cpu().numpy())
        finally:
            self.model.train(was_training)

        widths = {
            "mu": self.model.z_dim,
            "logvar": self.model.z_dim,
            "probs": self.model.n_categories,
        }

        return {
            name: (
                np.concatenate(values, axis=0)
                if values
                else np.zeros((0, widths[name]), dtype=np.float32)
            )
            for name, values in outputs.items()
        }
