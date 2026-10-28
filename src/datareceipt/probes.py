"""Tiny cross-validated probes used to ask: "can this signal alone predict
the label?"

Both probes are deliberately weak learners implemented in NumPy. A strong
accuracy from a weak learner is the point: if a nearest-centroid classifier
can recover the label from border pixels alone, the dataset is carrying the
label in its backgrounds, and a real model will find that shortcut too.

All folds are deterministic stratified round-robin folds, so a receipt is
reproducible byte-for-byte on the same data.
"""

from __future__ import annotations

import numpy as np

CAUTION_LIFT = 0.10
FAIL_LIFT = 0.25


def stratified_folds(labels: list[str], n_splits: int = 3) -> list[list[int]]:
    """Deterministic stratified folds: within each class (in sorted order),
    samples are dealt round-robin into the folds in dataset order."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    by_class: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)
    for label in sorted(by_class):
        for position, idx in enumerate(by_class[label]):
            folds[position % n_splits].append(idx)
    return [sorted(fold) for fold in folds]


def majority_rate(labels: list[str]) -> float:
    """Accuracy of always predicting the most common label ("chance")."""
    if not labels:
        return 0.0
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.values()) / len(labels)


def _usable_classes(labels: list[str], n_splits: int) -> list[str]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return sorted(label for label, count in counts.items() if count >= n_splits)


def nearest_centroid_cv_accuracy(
    features: np.ndarray, labels: list[str], n_splits: int = 3
) -> tuple[float, list[str]]:
    """Cross-validated accuracy of a standardized nearest-centroid probe.

    Returns ``(accuracy, classes_used)``. Classes with fewer samples than
    folds are excluded, because they cannot appear in every training fold.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(labels):
        raise ValueError("features must be a 2D array with one row per label")
    classes = _usable_classes(labels, n_splits)
    if len(classes) < 2:
        return 0.0, classes
    keep = [i for i, label in enumerate(labels) if label in classes]
    x = features[keep]
    y = [labels[i] for i in keep]
    class_index = {label: k for k, label in enumerate(classes)}
    y_idx = np.array([class_index[label] for label in y])

    folds = stratified_folds(y, n_splits)
    correct = 0
    total = 0
    for fold in folds:
        test_idx = np.array(fold, dtype=int)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        x_train, y_train = x[train_mask], y_idx[train_mask]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std < 1e-8] = 1.0
        z_train = (x_train - mean) / std
        z_test = (x[test_idx] - mean) / std
        centroids = np.stack(
            [z_train[y_train == k].mean(axis=0) for k in range(len(classes))]
        )
        distances = ((z_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        predictions = distances.argmin(axis=1)
        correct += int((predictions == y_idx[test_idx]).sum())
        total += len(test_idx)
    return (correct / total if total else 0.0), classes


def categorical_naive_bayes_cv_accuracy(
    rows: list[dict[str, str]], labels: list[str], n_splits: int = 3
) -> tuple[float, list[str]]:
    """Cross-validated accuracy of a categorical naive Bayes probe over
    string-valued features (one dict per sample)."""
    if len(rows) != len(labels):
        raise ValueError("rows and labels must have the same length")
    classes = _usable_classes(labels, n_splits)
    if len(classes) < 2:
        return 0.0, classes
    keep = [i for i, label in enumerate(labels) if label in classes]
    kept_rows = [rows[i] for i in keep]
    y = [labels[i] for i in keep]
    folds = stratified_folds(y, n_splits)

    feature_names = sorted({name for row in kept_rows for name in row})
    correct = 0
    total = 0
    for fold in folds:
        test_set = set(fold)
        train_idx = [i for i in range(len(y)) if i not in test_set]
        class_counts = {label: 0 for label in classes}
        value_counts: dict[str, dict[str, dict[str, int]]] = {
            name: {label: {} for label in classes} for name in feature_names
        }
        vocab: dict[str, set[str]] = {name: set() for name in feature_names}
        for i in train_idx:
            label = y[i]
            class_counts[label] += 1
            for name in feature_names:
                value = kept_rows[i].get(name, "")
                vocab[name].add(value)
                bucket = value_counts[name][label]
                bucket[value] = bucket.get(value, 0) + 1
        n_train = len(train_idx)
        for i in fold:
            best_label, best_score = classes[0], -np.inf
            for label in classes:
                score = np.log((class_counts[label] + 1) / (n_train + len(classes)))
                for name in feature_names:
                    value = kept_rows[i].get(name, "")
                    count = value_counts[name][label].get(value, 0)
                    score += np.log((count + 1) / (class_counts[label] + len(vocab[name]) + 1))
                if score > best_score:
                    best_label, best_score = label, score
            correct += int(best_label == y[i])
            total += 1
    return (correct / total if total else 0.0), classes


def lift_status(lift: float, caution: float = CAUTION_LIFT, fail: float = FAIL_LIFT) -> str:
    """Map a probe lift over chance to a check status."""
    if lift >= fail:
        return "fail"
    if lift >= caution:
        return "caution"
    return "pass"
