"""Downstream scores of one fold. Labels are used here and only here, after the VAE is trained.

Probe: StandardScaler + logistic regression (C = 1, balanced classes), the probe of CAD_VAE_CRICKET.
It is fitted on the latents of the training crickets and scored on crickets the VAE never saw.

Features ("branch" column, as in CAD_VAE_CRICKET):
    mu              posterior mean of z (the feature CAD-VAE uses)
    z               a sample from q(z|x): what the latent holds once its noise is counted
    c, muc          the category code (one-hot arg-max), alone and next to mu (JointVAE only)
    ref_pca10       10 principal components of the window, fitted on the training crickets
    ref_population  the population label alone   (traits only)
    ref_island      the island label alone       (traits only)
The three reference rows say how much of a score is population structure: any feature that only
knows which population a cricket is from scores as well as ref_population.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data import MISSING

TRAITS = ("venation", "shape")
GROUPS = ("population", "region", "island")
REFERENCE_LABELS = ("population", "island")
SPLITS = ("train", "val", "test")
MIN_PAIRS = 20  # a population is listed on its own when it has this many same-fold pairs


def make_classifier(C, class_weight, seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegression(
                max_iter=5000,
                solver="lbfgs",  # L2 penalty (the default)
                C=C,
                class_weight=class_weight,
                random_state=seed,
            ),
        ),
    ])


def macro_ovr_auc(y_true, probabilities, n_classes):
    class_aucs = []

    for class_id in range(n_classes):
        binary_true = (y_true == class_id).astype(int)

        if len(np.unique(binary_true)) == 2:
            class_aucs.append(roc_auc_score(binary_true, probabilities[:, class_id]))

    return float(np.mean(class_aucs)) if class_aucs else np.nan


def evaluate_probe(x_train, y_train, x_test, y_test, n_classes, C=1.0,
                   class_weight="balanced", seed=1024):
    classifier = make_classifier(C, class_weight, seed)
    classifier.fit(x_train, y_train)

    prediction = classifier.predict(x_test)
    fitted_classes = classifier.named_steps["logreg"].classes_.astype(int)

    probabilities = np.zeros((len(y_test), n_classes), dtype=float)
    probabilities[:, fitted_classes] = classifier.predict_proba(x_test)

    metrics = {
        "acc": accuracy_score(y_test, prediction),
        "balanced_acc": balanced_accuracy_score(y_test, prediction),
        "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
        "auc": macro_ovr_auc(y_test, probabilities, n_classes),
    }

    return metrics, prediction, probabilities


def principal_components(data, n_components=10):
    """PCs of the mean-imputed, centred dosage; mean and axes from the training crickets only."""
    def dosage(split_name):
        x = getattr(data, f"x_{split_name}").astype(np.float32)
        x[x == MISSING] = np.nan
        return x

    train = dosage("train")
    mean = np.nan_to_num(np.nanmean(train, axis=0), nan=0.0)

    def centred(x):
        return np.where(np.isnan(x), 0.0, x - mean)

    _, _, axes = np.linalg.svd(centred(train), full_matrices=False)
    axes = axes[:n_components]

    return {split_name: centred(dosage(split_name)) @ axes.T for split_name in SPLITS}


def build_features(data, latents, seed, n_draws=10):
    """Feature sets of one fold: name -> split -> list of arrays (one per draw)."""
    features = {"mu": {s: [latents[s]["mu"]] for s in SPLITS}}

    rng = np.random.default_rng(seed)
    features["z"] = {
        s: [
            latents[s]["mu"]
            + np.exp(0.5 * latents[s]["logvar"]) * rng.standard_normal(latents[s]["mu"].shape)
            for _ in range(n_draws)
        ]
        for s in SPLITS
    }

    if "probs" in latents["train"]:
        n_categories = latents["train"]["probs"].shape[1]
        one_hot = {
            s: np.eye(n_categories)[latents[s]["probs"].argmax(-1).astype(int)]
            if len(latents[s]["probs"]) else np.zeros((0, n_categories))
            for s in SPLITS
        }
        features["c"] = {s: [one_hot[s]] for s in SPLITS}
        features["muc"] = {
            s: [np.concatenate([latents[s]["mu"], one_hot[s]], axis=1)] for s in SPLITS
        }

    pcs = principal_components(data)
    features["ref_pca10"] = {s: [pcs[s]] for s in SPLITS}

    for label in REFERENCE_LABELS:
        classes = np.unique(data.all_labels(label))
        features[f"ref_{label}"] = {
            s: [(data.labels(s, label)[:, None] == classes[None, :]).astype(float)]
            for s in SPLITS
        }

    return features


def encode_tasks(data):
    """Integer labels of every task for every split, and the class names."""
    tasks, mapping_rows = {}, []

    for task in TRAITS + GROUPS:
        encoder = LabelEncoder()
        encoder.fit(data.all_labels(task))

        tasks[task] = {
            "n_classes": len(encoder.classes_),
            **{s: encoder.transform(data.labels(s, task)) for s in SPLITS},
        }

        for class_id, class_name in enumerate(encoder.classes_):
            mapping_rows.append({"task": task, "encoded_id": class_id, "class_name": class_name})

    return tasks, pd.DataFrame(mapping_rows)


def evaluate_downstream(data, features, fold_id, fold_dir, eval_split="test"):
    """Fit every probe on the training crickets and score it on eval_split.

    eval_split="test": the reported numbers. eval_split="val": the inner validation crickets,
    for choosing settings without touching the test fold.
    """
    if len(getattr(data, f"{eval_split}_indices")) == 0:
        return pd.DataFrame()

    tasks, mapping = encode_tasks(data)

    metric_rows = []
    prediction_table = {
        "sample": getattr(data, f"sample_ids_{eval_split}"),
        "fold_id": fold_id,
    }

    for task, encoded in tasks.items():
        y_train, y_eval, n_classes = encoded["train"], encoded[eval_split], encoded["n_classes"]
        prediction_table[f"{task}_true"] = y_eval

        for branch, by_split in features.items():
            if task in GROUPS and branch in ("ref_population", "ref_island"):
                continue

            draws = [
                evaluate_probe(x_train, y_train, x_eval, y_eval, n_classes, seed=data.cfg.seed)
                for x_train, x_eval in zip(by_split["train"], by_split[eval_split])
            ]

            metrics = {
                name: float(np.nanmean([d[0][name] for d in draws]))
                for name in ("acc", "balanced_acc", "macro_f1", "auc")
            }
            probability = np.mean([d[2] for d in draws], axis=0)

            metric_rows.append({
                "fold_id": fold_id,
                "task": task,
                "branch": branch,
                "n_classes": n_classes,
                "chance_balanced_acc": 1.0 / n_classes,
                **metrics,
            })

            prefix = f"{task}_{branch}"
            prediction_table[f"{prefix}_prediction"] = probability.argmax(-1)

            for class_id in range(n_classes):
                prediction_table[f"{prefix}_probability_{class_id}"] = probability[:, class_id]

    metrics = pd.DataFrame(metric_rows)
    suffix = "" if eval_split == "test" else f"_{eval_split}"

    metrics.to_csv(os.path.join(fold_dir, f"downstream_metrics{suffix}.csv"), index=False)

    if eval_split == "test":
        mapping.to_csv(os.path.join(fold_dir, "class_mapping.csv"), index=False)
        pd.DataFrame(prediction_table).to_csv(
            os.path.join(fold_dir, "downstream_predictions.csv"),
            index=False,
        )

    return metrics


def cluster_scores(clusters, classes):
    """Accuracy after the best one-to-one matching of clusters to classes, NMI and ARI."""
    size = int(max(clusters.max(), classes.max())) + 1
    counts = np.zeros((size, size))
    np.add.at(counts, (clusters, classes), 1)
    rows, columns = linear_sum_assignment(-counts)

    return {
        "acc": float(counts[rows, columns].sum() / len(clusters)),
        "nmi": float(normalized_mutual_info_score(classes, clusters)),
        "ari": float(adjusted_rand_score(classes, clusters)),
    }


def evaluate_clustering(data, latents, features, fold_id, fold_dir, eval_split="test"):
    """Recovery of island, region and population without a probe, on the crickets of eval_split.

    code:    the category of the JointVAE (arg-max), compared with the labels.
    kmeans:  k-means with as many clusters as classes, fitted on the training crickets.
    """
    if len(getattr(data, f"{eval_split}_indices")) == 0:
        return pd.DataFrame()

    tasks, _ = encode_tasks(data)
    rows = []

    for task in GROUPS:
        y_test, n_classes = tasks[task][eval_split], tasks[task]["n_classes"]

        if "probs" in latents[eval_split]:
            codes = latents[eval_split]["probs"].argmax(-1).astype(int)
            rows.append({
                "fold_id": fold_id, "task": task, "method": "code",
                "clusters_used": len(np.unique(codes)), **cluster_scores(codes, y_test),
            })

        for name in ("mu", "ref_pca10"):
            scaler = StandardScaler().fit(features[name]["train"][0])
            kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=data.cfg.seed)
            kmeans.fit(scaler.transform(features[name]["train"][0]))
            clusters = kmeans.predict(scaler.transform(features[name][eval_split][0]))
            rows.append({
                "fold_id": fold_id, "task": task, "method": f"kmeans_{name}",
                "clusters_used": len(np.unique(clusters)), **cluster_scores(clusters, y_test),
            })

    result = pd.DataFrame(rows)
    suffix = "" if eval_split == "test" else f"_{eval_split}"
    result.to_csv(os.path.join(fold_dir, f"clustering_metrics{suffix}.csv"), index=False)

    return result


def summarize(results, keys, metrics):
    """Mean, sample standard deviation and count over the folds."""
    rows = []

    for values, frame in results.groupby(list(keys), sort=False):
        row = dict(zip(keys, values))

        for metric in metrics:
            numbers = frame[metric].to_numpy(dtype=float)
            row[f"cv_{metric}_mean"] = np.nanmean(numbers)
            row[f"cv_{metric}_std"] = np.nanstd(numbers, ddof=1)
            row[f"cv_{metric}_count"] = int(np.isfinite(numbers).sum())

        rows.append(row)

    return pd.DataFrame(rows)


def stratified_auc(score, y, strata):
    """AUC from the pairs of one positive and one negative cricket of the same stratum.

    Returns (AUC, number of pairs). Ties count one half; scores closer than 1e-9 are ties
    (equal inputs can differ in the last digit after the matrix product of the probe).
    """
    concordant, total = 0.0, 0

    for stratum in np.unique(strata):
        member = strata == stratum
        positive = score[member & (y == 1)]
        negative = score[member & (y == 0)]

        if len(positive) and len(negative):
            difference = positive[:, None] - negative[None, :]
            tied = np.abs(difference) <= 1e-9
            concordant += ((difference > 0) & ~tied).sum() + 0.5 * tied.sum()
            total += difference.size

    return (float(concordant / total) if total else np.nan), int(total)


def pooled_trait_summary(save_dir, n_folds, metadata_all):
    """Trait scores from the out-of-fold predictions of all crickets together.

    pooled_auc             one AUC over all crickets (each scored by the fold that left it out)
    within_island_auc      AUC from pairs of crickets of the same island and the same fold
    within_population_auc  AUC from pairs of crickets of the same population and the same fold
    A trait signal that is only population structure gives a high pooled AUC and 0.5 inside the
    populations.

    Pairs are taken inside one fold because scores of different folds come from different models
    and probes. Comparing them across folds would also bias the result downwards: a population
    with more long-veined crickets in the training part has fewer of them in the test part, so a
    probe that has only learnt the population frequencies ranks the test crickets the wrong way
    round (the label-only reference then scores about 0.3 instead of 0.5).
    """
    predictions = pd.concat(
        [
            pd.read_csv(os.path.join(save_dir, f"fold_{fold_id}", "downstream_predictions.csv"))
            for fold_id in range(1, n_folds + 1)
        ],
        ignore_index=True,
    )

    lookup = metadata_all.assign(sample=metadata_all["sample"].astype(str)).set_index("sample")
    samples = predictions["sample"].astype(str)
    population = np.asarray(lookup.loc[samples, "population"], dtype=str)
    island = np.asarray(lookup.loc[samples, "island"], dtype=str)
    fold = predictions["fold_id"].to_numpy().astype(str)

    in_island = np.char.add(np.char.add(fold, "|"), island)
    in_population = np.char.add(np.char.add(fold, "|"), population)

    rows = []

    for task in TRAITS:
        y = predictions[f"{task}_true"].to_numpy().astype(int)
        suffix = "_probability_1"

        for column in predictions.columns:
            if not (column.startswith(f"{task}_") and column.endswith(suffix)):
                continue

            branch = column[len(task) + 1:-len(suffix)]
            score = predictions[column].to_numpy()

            island_auc, island_pairs = stratified_auc(score, y, in_island)
            population_auc, population_pairs = stratified_auc(score, y, in_population)

            detail = {}

            for name in np.unique(population):
                member = population == name
                value, pairs = stratified_auc(score[member], y[member], fold[member])

                if pairs >= MIN_PAIRS:
                    detail[str(name)] = round(value, 3)

            rows.append({
                "task": task,
                "branch": branch,
                "n": len(y),
                "pooled_auc": float(roc_auc_score(y, score)),
                "within_island_auc": island_auc,
                "within_island_pairs": island_pairs,
                "within_population_auc": population_auc,
                "within_population_pairs": population_pairs,
                "population_auc_detail": json.dumps(detail),
            })

    return pd.DataFrame(rows)
