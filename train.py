"""Run full-pipeline JointVAE cross-validation for one LD window.

For every outer fold: train the JointVAE on the training crickets (no labels), export the latents,
then fit the downstream probes on the training latents and score them on the test crickets, which
neither the VAE nor the probe has seen. Folds and output layout follow CAD_VAE_CRICKET/train.py.
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from data import DataConfig, LDWindowDataModule
from downstream import (
    SPLITS,
    build_features,
    evaluate_clustering,
    evaluate_downstream,
    pooled_trait_summary,
    summarize,
)
from model import JointVAE, ModelConfig
from trainer import JointVAETrainer, TrainConfig

PROBE_METRICS = ("acc", "balanced_acc", "macro_f1", "auc")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")


def save_json(values, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)


def export_latents(data, trainer, fold_dir):
    """Export the posterior of every split in stored sample order."""
    latents = {}

    for split_name in SPLITS:
        encoded = trainer.encode_dataset(getattr(data, f"x_{split_name}"))
        latents[split_name] = encoded

        metadata = getattr(data, f"{split_name}_metadata")

        np.savez_compressed(
            os.path.join(fold_dir, f"{split_name}_latents.npz"),
            mu=encoded["mu"],
            logvar=encoded["logvar"],
            probs=encoded["probs"],
            code=encoded["probs"].argmax(-1) if len(encoded["probs"]) else np.zeros(0, dtype=int),
            morph=metadata["morph"].to_numpy(dtype=int),
            sample_ids=np.asarray(getattr(data, f"sample_ids_{split_name}"), dtype=str),
            indices=getattr(data, f"{split_name}_indices"),
        )

        metadata.to_csv(os.path.join(fold_dir, f"{split_name}_metadata.csv"), index=False)

    return latents


def plot_training_losses(history, path, title):
    """Save the loss curves of one fold (skipped when matplotlib is not installed)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    panels = (
        ("loss", "Objective"),
        ("recon", "Reconstruction (nats per cricket)"),
        ("kl_z", "KL of z"),
        ("kl_c", "KL of c"),
    )

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), sharex=True)

    for ax, (metric, panel_title) in zip(axes, panels):
        for prefix, label, color in (("train", "Train", "#0072B2"), ("val", "Validation", "#D55E00")):
            column = f"{prefix}_{metric}"

            if column in history.columns:
                ax.plot(history["epoch"], history[column], label=label, color=color, linewidth=1.2)

        if metric in ("kl_z", "kl_c"):
            ax.plot(history["epoch"], history[f"capacity_{metric[-1]}"], label="Capacity",
                    color="#999999", linewidth=1.0, linestyle="--")

        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run five-fold JointVAE training for one LD window."
    )

    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ld_block_pkl", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--block_id", type=int, default=0)

    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--use_ext_window", dest="use_ext_window", action="store_true")
    window_group.add_argument("--use_core_window", dest="use_ext_window", action="store_false")
    parser.set_defaults(use_ext_window=True)

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--inner_val_fraction", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=1024,
                        help="Seed of the folds (and of the model when --model_seed is not given).")
    parser.add_argument("--model_seed", type=int, default=None,
                        help="Seed of the model initialisation and the training noise.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default=None)

    # Model.
    parser.add_argument("--n_categories", type=int, default=3)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dims", default="512,128",
                        help="Widths of the encoder MLP (the decoder mirrors them).")

    # Loss weights.
    parser.add_argument("--gamma", type=float, default=1000.0,
                        help="Weight of both capacity terms (z and c).")
    parser.add_argument("--capacity_z", type=float, default=1.0,
                        help="Final capacity of z in nats. The capacity of c ends at log(K).")
    parser.add_argument("--lambda_marginal", type=float, default=0.0,
                        help="Weight of the marginal-prior term (0 = Dupont's JointVAE).")
    parser.add_argument("--select", choices=["last", "best_val"], default="last")

    parser.add_argument("--n_draws", type=int, default=10,
                        help="Samples from q(z|x) for the 'z' features.")

    return parser.parse_args()


def main(args):
    model_seed = args.seed if args.model_seed is None else args.model_seed

    set_seed(model_seed)
    os.makedirs(args.save_dir, exist_ok=True)

    data_cfg = DataConfig(
        out_dir=args.out_dir,
        ld_block_pkl=args.ld_block_pkl,
        block_id=args.block_id,
        use_ext_window=args.use_ext_window,
        seed=args.seed,
        n_splits=args.n_splits,
        inner_val_fraction=args.inner_val_fraction,
    )

    model_cfg = ModelConfig(
        z_dim=args.z_dim,
        n_categories=args.n_categories,
        dropout=args.dropout,
        hidden_dims=tuple(int(width) for width in args.hidden_dims.split(",")),
    )

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        gamma_z=args.gamma,
        capacity_z=args.capacity_z,
        lambda_marginal=args.lambda_marginal,
        select=args.select,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    save_json(
        {
            "data": asdict(data_cfg),
            "model": asdict(model_cfg),
            "training": asdict(train_cfg),
            "device": device,
            "model_seed": model_seed,
            "downstream": {
                "classifier": "StandardScaler + LogisticRegression",
                "C": 1.0,
                "solver": "lbfgs",
                "penalty": "l2",
                "max_iter": 5000,
                "class_weight": "balanced",
                "random_state": args.seed,
                "fitted_on": "latents of the training crickets of the fold",
                "tasks": ["venation", "shape", "population", "region", "island"],
                "features": "mu, z (mean over draws), c, muc, ref_pca10, ref_population, ref_island",
                "n_draws": args.n_draws,
            },
        },
        os.path.join(args.save_dir, "config.json"),
    )

    data = LDWindowDataModule(data_cfg)
    data.setup()

    folds = data.make_cv_splits()

    data.metadata_all.to_csv(os.path.join(args.save_dir, "sample_pool_metadata.csv"), index=False)

    vae_rows, test_results, val_results = [], [], []
    clustering_results, clustering_val_results = [], []

    for fold_id, (train_indices, val_indices, test_indices) in enumerate(folds, start=1):
        fold_dir = os.path.join(args.save_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        data.set_split(train_indices, val_indices, test_indices, fold_id=fold_id)

        for split_name in SPLITS:
            np.save(
                os.path.join(fold_dir, f"{split_name}_indices.npy"),
                getattr(data, f"{split_name}_indices"),
            )

        save_json(data.block_info, os.path.join(fold_dir, "block_info.json"))

        # Start each fold with the same seed and a fresh model and optimizer.

        set_seed(model_seed)

        model = JointVAE(data.input_dim, model_cfg)
        trainer = JointVAETrainer(model=model, cfg=train_cfg, device=device)

        fit_start = time.time()
        trainer.fit(data.x_train, data.x_val)
        fit_seconds = time.time() - fit_start
        trainer.restore_best_model()

        history = pd.DataFrame(trainer.history)
        history.to_csv(os.path.join(fold_dir, "training_log.csv"), index=False)

        plot_training_losses(
            history,
            os.path.join(fold_dir, "training_losses.png"),
            title=(
                f"JointVAE | block_{args.block_id} | fold_{fold_id} | K={args.n_categories} | "
                f"gamma={args.gamma:g} | C_z={args.capacity_z:g} | "
                f"lambda={args.lambda_marginal:g}"
            ),
        )

        vae_row = {
            "fold_id": fold_id,
            "best_epoch": trainer.best_epoch,
            "best_val_loss": trainer.best_val_loss,
            "n_snps_used": data.x_train.shape[1],
            "fit_seconds": fit_seconds,
            **trainer.evaluate(data.x_test),
            **{
                f"val_{name}": value
                for name, value in trainer.evaluate(data.x_val).items()
            },
        }

        vae_rows.append(vae_row)
        save_json(vae_row, os.path.join(fold_dir, "final_test_metrics.json"))

        latents = export_latents(data, trainer, fold_dir)

        # From here on labels are used; the model is no longer changed.
        features = build_features(data, latents, seed=model_seed + fold_id, n_draws=args.n_draws)

        test_results.append(evaluate_downstream(data, features, fold_id, fold_dir, "test"))
        val_results.append(evaluate_downstream(data, features, fold_id, fold_dir, "val"))
        clustering_results.append(
            evaluate_clustering(data, latents, features, fold_id, fold_dir, "test")
        )
        clustering_val_results.append(
            evaluate_clustering(data, latents, features, fold_id, fold_dir, "val")
        )

        del trainer, model

    vae_results = pd.DataFrame(vae_rows)
    vae_results.to_csv(os.path.join(args.save_dir, "vae_test_metrics_all_folds.csv"), index=False)

    vae_summary = (
        vae_results.drop(columns=["fold_id", "best_epoch", "best_val_loss"])
        .agg(["mean", "std", "count"])
        .T
    )
    vae_summary.index.name = "metric"
    vae_summary.to_csv(os.path.join(args.save_dir, "vae_test_summary.csv"))

    clustering = pd.concat(clustering_results, ignore_index=True)
    clustering.to_csv(os.path.join(args.save_dir, "clustering_metrics_all_folds.csv"), index=False)
    summarize(clustering, ("task", "method"), ("acc", "nmi", "ari", "clusters_used")).to_csv(
        os.path.join(args.save_dir, "clustering_summary.csv"),
        index=False,
    )

    val_results = [frame for frame in val_results if len(frame)]

    if val_results:
        summarize(
            pd.concat(clustering_val_results, ignore_index=True),
            ("task", "method"),
            ("acc", "nmi", "ari", "clusters_used"),
        ).to_csv(os.path.join(args.save_dir, "clustering_summary_val.csv"), index=False)

        validation = pd.concat(val_results, ignore_index=True)
        validation.to_csv(
            os.path.join(args.save_dir, "downstream_metrics_val_all_folds.csv"),
            index=False,
        )
        summarize(validation, ("task", "branch"), PROBE_METRICS).to_csv(
            os.path.join(args.save_dir, "downstream_summary_val.csv"),
            index=False,
        )

    pooled_trait_summary(args.save_dir, len(folds), data.metadata_all).to_csv(
        os.path.join(args.save_dir, "downstream_pooled.csv"),
        index=False,
    )

    downstream_results = pd.concat(test_results, ignore_index=True)
    downstream_results.to_csv(
        os.path.join(args.save_dir, "downstream_metrics_all_folds.csv"),
        index=False,
    )

    # Written last: the Slurm scripts take this file as the mark of a finished window.
    summarize(downstream_results, ("task", "branch"), PROBE_METRICS).to_csv(
        os.path.join(args.save_dir, "downstream_summary.csv"),
        index=False,
    )

    print(f"Saved results to: {args.save_dir}")


if __name__ == "__main__":
    main(parse_args())
