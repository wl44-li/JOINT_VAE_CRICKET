"""Load LD-window genotypes for full-pipeline cross-validation.

Same input files, same LD windows and the same folds as CAD_VAE_CRICKET (data.py there): the stored
train / hold-out partitions are merged (297 crickets), split into outer folds with StratifiedKFold in
sample-id order, stratified by the 4-class morph label, and a share of every training part is set
aside as inner validation. With the same seed the folds are identical, cricket for cricket.

Labels are loaded here but never given to the model: they are used for the stratification of the
folds and, after training, for the downstream scores.
"""

import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import zarr
from sklearn.model_selection import StratifiedKFold, train_test_split


MISSING = 3  # dosage code of a missing call

ISLAND_NAMES = ("Hawaii/Hilo", "Kauai", "Oahu")


@dataclass
class DataConfig:
    out_dir: str
    ld_block_pkl: str

    block_id: int = 0
    use_ext_window: bool = True

    seed: int = 1024

    n_splits: int = 5
    inner_val_fraction: float = 0.125


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def population_to_island(population):
    """Map population names to the three islands."""
    population = str(population).strip()

    if population.startswith(("Hilo", "Hawaii")):
        return "Hawaii/Hilo"

    if population.startswith("Kauai"):
        return "Kauai"

    if population.startswith("Oahu"):
        return "Oahu"

    raise ValueError(f"Cannot derive island from population: {population}")


def channels_to_dosage(x):
    """[n, snps, 2] (ALT presence, REF presence) -> uint8 dosage; a missing call [0, 0] -> 3."""
    alt, ref = x[..., 0], x[..., 1]
    dosage = (alt + (1 - ref)).astype(np.uint8)  # (0,1) -> 0, (1,1) -> 1, (1,0) -> 2
    dosage[(alt == 0) & (ref == 0)] = MISSING
    return dosage


def validate_partition(train_indices, val_indices, test_indices, n_samples):
    combined = np.concatenate([train_indices, val_indices, test_indices])

    if not np.array_equal(np.sort(combined), np.arange(n_samples)):
        raise ValueError("Train/val/test must be a full partition of all samples.")


class LDWindowDataModule:
    """Load one LD window and select samples for each CV fold."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        x_train_zarr = zarr.open(os.path.join(self.cfg.out_dir, "x_train.zarr"), mode="r")
        x_holdout_zarr = zarr.open(os.path.join(self.cfg.out_dir, "x_test.zarr"), mode="r")

        original_y_train = np.asarray(
            load_pickle(os.path.join(self.cfg.out_dir, "y_train.pkl")), dtype=int
        ).reshape(-1)
        original_y_holdout = np.asarray(
            load_pickle(os.path.join(self.cfg.out_dir, "y_test.pkl")), dtype=int
        ).reshape(-1)

        train_metadata = pd.read_pickle(
            os.path.join(self.cfg.out_dir, "train_metadata.pkl")
        ).reset_index(drop=True)
        holdout_metadata = pd.read_pickle(
            os.path.join(self.cfg.out_dir, "test_metadata.pkl")
        ).reset_index(drop=True)

        if not (x_train_zarr.shape[0] == len(original_y_train) == len(train_metadata)):
            raise ValueError("Training genotype/label/metadata lengths differ.")

        if not (x_holdout_zarr.shape[0] == len(original_y_holdout) == len(holdout_metadata)):
            raise ValueError("Holdout genotype/label/metadata lengths differ.")

        block = load_pickle(self.cfg.ld_block_pkl)["merged_blocks"].iloc[self.cfg.block_id]

        if self.cfg.use_ext_window:
            start = int(block["ext_start_idx_full"])
            end = int(block["ext_end_idx_full"])
            n_snps = int(block["ext_n_snps"])
            window_type = "extended"
        else:
            start = int(block["core_start_idx_full"])
            end = int(block["core_end_idx_full"])
            n_snps = int(block["core_n_snps"])
            window_type = "core"

        # Merge the stored partitions before making new CV folds.
        channels = np.concatenate(
            [
                np.asarray(x_train_zarr[:, start:end + 1, :]),
                np.asarray(x_holdout_zarr[:, start:end + 1, :]),
            ],
            axis=0,
        )
        self.x_all = np.ascontiguousarray(channels_to_dosage(channels))

        if self.x_all.shape[1] != n_snps:
            raise ValueError(f"{self.x_all.shape[1]} SNPs read, {n_snps} expected.")

        self.original_y_all = np.concatenate([original_y_train, original_y_holdout], axis=0)

        self.metadata_all = pd.concat([train_metadata, holdout_metadata], ignore_index=True)
        self.metadata_all["island"] = self.metadata_all["population"].map(population_to_island)

        self.sample_ids_all = np.asarray(
            [str(value) for value in self.metadata_all["sample"]],
            dtype=str,
        )

        if pd.Series(self.sample_ids_all).duplicated().any():
            raise ValueError("Duplicate sample IDs in the merged dataset.")

        self.n_snps = n_snps
        self.input_dim = 2 * n_snps
        self.missing_share_all = (self.x_all == MISSING).mean(axis=1)

        self._window_info = {
            "block_id": self.cfg.block_id,
            "window_type": window_type,
            "start": start,
            "end": end,
            "n_snps": n_snps,
            "input_dim": self.input_dim,
            "missing_share": float(self.missing_share_all.mean()),
            "stratification_source": "original y pickle files",
            "n_splits": self.cfg.n_splits,
            "inner_val_fraction": self.cfg.inner_val_fraction,
        }

    def make_cv_splits(self):
        """Return global (train, val, test) indices for each fold."""
        ordered_indices = np.argsort(self.sample_ids_all, kind="stable")
        strata = self.original_y_all[ordered_indices]

        splitter = StratifiedKFold(
            n_splits=self.cfg.n_splits,
            shuffle=True,
            random_state=self.cfg.seed,
        )

        folds = []

        for train_val_positions, test_positions in splitter.split(ordered_indices, strata):
            train_val_indices = ordered_indices[train_val_positions]
            test_indices = ordered_indices[test_positions]

            if self.cfg.inner_val_fraction > 0:
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=self.cfg.inner_val_fraction,
                    random_state=self.cfg.seed,
                    stratify=self.original_y_all[train_val_indices],
                )
            else:  # no inner validation: train on everything outside the test fold
                train_indices = train_val_indices
                val_indices = np.empty(0, dtype=train_val_indices.dtype)

            folds.append((train_indices, val_indices, test_indices))

        return folds

    def set_split(self, train_indices, val_indices, test_indices, fold_id):
        """Select the train/val/test samples for one fold."""
        train_indices = np.asarray(train_indices)
        val_indices = np.asarray(val_indices)
        test_indices = np.asarray(test_indices)

        validate_partition(train_indices, val_indices, test_indices, len(self.x_all))

        self.fold_id = fold_id

        for split_name, indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ):
            setattr(self, f"{split_name}_indices", indices.copy())
            setattr(self, f"x_{split_name}", self.x_all[indices])
            setattr(
                self,
                f"{split_name}_metadata",
                self.metadata_all.iloc[indices].reset_index(drop=True),
            )
            setattr(self, f"sample_ids_{split_name}", self.sample_ids_all[indices])

        self.block_info = {
            **self._window_info,
            "fold_id": fold_id,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
        }

        print(self.block_info)

    def all_labels(self, column):
        """Labels of all crickets as a numpy array of strings (used only after training)."""
        return np.asarray(
            [str(value).strip() for value in self.metadata_all[column]],
            dtype=str,
        )

    def labels(self, split_name, column):
        """Labels of one split as a numpy array of strings (used only after training)."""
        return self.all_labels(column)[getattr(self, f"{split_name}_indices")]
