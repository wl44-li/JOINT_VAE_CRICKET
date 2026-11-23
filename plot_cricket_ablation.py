#!/usr/bin/env python3
"""Plot the ablation runs against one setting.

By default only numbers that do not touch the test folds are shown: label-free numbers of the
model and probe / clustering scores on the inner validation crickets. Choose settings from these.
--show_test adds the test-fold scores, for a look after the choice has been made.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

SETTINGS = ("n_categories", "gamma_z", "capacity_z", "lambda_marginal", "dropout")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Folder with the ablation runs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vary", required=True, choices=SETTINGS)
    parser.add_argument("--show_test", action="store_true")
    return parser.parse_args()


def lookup(frame, column, **where):
    """One value of a summary table, or None when the row is missing."""
    for key, value in where.items():
        frame = frame[frame[key] == value]
    return float(frame[column].iloc[0]) if len(frame) else None


def read_run(run_dir, show_test):
    config = json.loads((run_dir / "config.json").read_text())
    scaffold = re.search(r"scaffold_(\d+)", config["data"]["out_dir"])

    row = {
        "window": f"{scaffold.group(1) if scaffold else '?'}:{config['data']['block_id']}",
        "n_categories": config["model"]["n_categories"],
        "dropout": config["model"]["dropout"],
        "gamma_z": config["training"]["gamma_z"],
        "capacity_z": config["training"]["capacity_z"],
        "lambda_marginal": config["training"]["lambda_marginal"],
    }

    vae = pd.read_csv(run_dir / "vae_test_metrics_all_folds.csv").mean(numeric_only=True)
    row["val: reconstruction per feature"] = vae.get("val_recon_per_feature")
    row["val: categories used"] = vae.get("val_categories_used")
    row["val: information in c (nats)"] = vae.get("val_category_information")
    row["val: active z dimensions"] = vae.get("val_n_active_z")

    splits = [("val", "_val")] + ([("test", "")] if show_test else [])

    for split_name, suffix in splits:
        probes = run_dir / f"downstream_summary{suffix}.csv"
        clusters = run_dir / f"clustering_summary{suffix}.csv"

        if probes.is_file():
            frame = pd.read_csv(probes)
            row[f"{split_name}: island from z (balanced acc.)"] = lookup(
                frame, "cv_balanced_acc_mean", task="island", branch="z")
            row[f"{split_name}: venation AUC from mu"] = lookup(
                frame, "cv_auc_mean", task="venation", branch="mu")
            row[f"{split_name}: shape AUC from mu"] = lookup(
                frame, "cv_auc_mean", task="shape", branch="mu")

        if clusters.is_file():
            frame = pd.read_csv(clusters)
            row[f"{split_name}: island recovered by c"] = lookup(
                frame, "cv_acc_mean", task="island", method="code")

    return row


def main():
    args = parse_args()
    root, output_dir = Path(args.root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        read_run(path.parent, args.show_test)
        for path in sorted(root.rglob("downstream_summary.csv"))
        if (path.parent / "config.json").is_file()
    ]

    if not runs:
        raise SystemExit(f"No finished runs under {root}")

    runs = pd.DataFrame(runs)
    metrics = [c for c in runs.columns if c not in SETTINGS and c != "window"]
    others = [s for s in SETTINGS if s != args.vary]

    # Mean over the windows for every setting.
    table = runs.groupby(list(SETTINGS), as_index=False)[metrics].mean()
    table.insert(len(SETTINGS), "windows", runs.groupby(list(SETTINGS)).size().to_numpy())
    table.to_csv(output_dir / f"ablation_table_{args.vary}.csv", index=False)

    n_columns = 4
    n_rows = -(-len(metrics) // n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4.2 * n_columns, 3.3 * n_rows), squeeze=False)

    for ax, metric in zip(axes.flat, metrics):
        for values, line in table.groupby(others):
            if line[args.vary].nunique() < 2:
                continue

            line = line.sort_values(args.vary)
            label = ", ".join(f"{name}={value:g}" for name, value in zip(others, values))
            ax.plot(line[args.vary].astype(str), line[metric], marker="o", linewidth=1.2, label=label)

        ax.set_title(metric, fontsize=9)
        ax.set_xlabel(args.vary)
        ax.grid(alpha=0.2)

    for ax in list(axes.flat)[len(metrics):]:
        fig.delaxes(ax)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7)

    fig.suptitle(
        f"JointVAE ablation over {args.vary}: mean of {runs['window'].nunique()} tuning windows",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_dir / f"ablation_{args.vary}.png", dpi=150)
    plt.close(fig)

    print(table.round(3).to_string(index=False))
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
