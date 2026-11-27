"""Compare exploration settings with the base setting, window by window.

Only numbers that do not touch the test folds are read: label-free numbers of the model and probe
scores on the inner validation crickets. For every setting the table gives the mean over windows
and the paired difference from the base setting with its standard error (same windows, same folds).
--show_test adds the test-fold scores, for the final check only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Folder with one sub-folder per setting.")
    parser.add_argument("--windows", required=True, help="File with 'scaffold block group' lines.")
    parser.add_argument("--base", default="base")
    parser.add_argument("--output", default=None)
    parser.add_argument("--show_test", action="store_true")
    return parser.parse_args()


def lookup(frame, column, **where):
    for key, value in where.items():
        frame = frame[frame[key] == value]
    return float(frame[column].iloc[0]) if len(frame) else np.nan


def read_run(run_dir, show_test):
    vae = pd.read_csv(run_dir / "vae_test_metrics_all_folds.csv").mean(numeric_only=True)
    row = {
        "snps used": vae.get("n_snps_used"),
        "fit seconds": vae.get("fit_seconds"),
        "best epoch": vae.get("best_epoch"),
        "val concordance": vae.get("val_concordance"),
        "val nll per snp": vae.get("val_recon_per_snp"),
        "val KL": vae.get("val_kl"),
    }

    for split, suffix in [("val", "_val")] + ([("test", "")] if show_test else []):
        probes = pd.read_csv(run_dir / f"downstream_summary{suffix}.csv")
        row[f"{split} venation AUC"] = lookup(probes, "cv_auc_mean", task="venation", branch="mu")
        row[f"{split} shape AUC"] = lookup(probes, "cv_auc_mean", task="shape", branch="mu")
        row[f"{split} island bacc"] = lookup(probes, "cv_balanced_acc_mean", task="island", branch="mu")
        row[f"{split} island bacc z"] = lookup(probes, "cv_balanced_acc_mean", task="island", branch="z")
        row[f"{split} population bacc"] = lookup(probes, "cv_balanced_acc_mean", task="population", branch="mu")

    if show_test:
        pooled = pd.read_csv(run_dir / "downstream_pooled.csv")
        for trait in ("venation", "shape"):
            row[f"test {trait} within-pop AUC"] = lookup(pooled, "within_population_auc", task=trait, branch="mu")
            row[f"test {trait} within-pop AUC pca10"] = lookup(pooled, "within_population_auc", task=trait, branch="ref_pca10")
            row[f"test {trait} AUC pca10"] = lookup(
                pd.read_csv(run_dir / "downstream_summary.csv"), "cv_auc_mean", task=trait, branch="ref_pca10")

    return row


def main():
    args = parse_args()
    root = Path(args.root)

    windows = [line.split() for line in Path(args.windows).read_text().splitlines() if line.strip()]
    rows = []

    for setting_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for scaffold, block, group in windows:
            run_dir = setting_dir / f"scaffold_{scaffold}" / f"block_{block}"

            if (run_dir / "downstream_summary.csv").is_file():
                rows.append({"setting": setting_dir.name, "window": f"{scaffold}:{block}", "group": group,
                             **read_run(run_dir, args.show_test)})

    runs = pd.DataFrame(rows)
    metrics = [c for c in runs.columns if c not in ("setting", "window", "group")]
    base = runs[runs["setting"] == args.base].set_index("window")[metrics]

    pd.set_option("display.width", 320, "display.max_columns", 60, "display.max_rows", 400)
    print(f"runs: {len(runs)}; settings: {runs['setting'].nunique()}; windows per setting:")
    print(runs.groupby("setting").size().to_string())

    def table(frame, title):
        means = frame.groupby("setting")[metrics].mean()
        print(f"\n## {title}: mean over windows")
        print(means.round(3).to_string())

        deltas, errors = {}, {}

        for setting, part in frame.groupby("setting"):
            part = part.set_index("window")[metrics]
            diff = (part - base.reindex(part.index)).dropna(how="all")
            deltas[setting] = diff.mean()
            errors[setting] = diff.std() / np.sqrt(max(len(diff), 1))

        deltas, errors = pd.DataFrame(deltas).T, pd.DataFrame(errors).T
        print(f"\n## {title}: paired difference from '{args.base}' (mean)")
        print(deltas.round(3).to_string())
        print(f"\n## {title}: standard error of that difference")
        print(errors.round(3).to_string())
        return means, deltas, errors

    means, deltas, errors = table(runs, "all windows")

    for group in ("peak", "tuning", "random"):
        part = runs[runs["group"] == group]
        if len(part):
            table(part, f"{group} windows")

    if args.output:
        runs.to_csv(args.output, index=False)
        out = means.add_prefix("mean ").join(deltas.add_prefix("delta ")).join(errors.add_prefix("se "))
        out.to_csv(str(args.output).replace(".csv", "_settings.csv"))


if __name__ == "__main__":
    main()
