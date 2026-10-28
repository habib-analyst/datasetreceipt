"""Metadata shortcut check.

The question: *can the label be predicted from everything except the
pixels?* Filenames, EXIF tags, image dimensions, and file sizes are all
things a dataset carries for free and a model (or a data pipeline) can end
up keying on. Classic real-world examples: one class scanned on one machine,
another class exported by different software, class A saved as PNG and
class B as JPEG.

Two views are reported:

* the strongest single metadata feature, measured by normalized mutual
  information with the label (fraction of label uncertainty it explains);
* a categorical naive Bayes probe over all metadata features together,
  cross-validated, compared against the majority-class chance rate.
"""

from __future__ import annotations

import math
import re

import numpy as np

from .probes import categorical_naive_bayes_cv_accuracy, lift_status, majority_rate

_TOKEN_SPLIT = re.compile(r"[^a-z]+")


def filename_prefix(name: str | None) -> str:
    """First alphabetic token of the file stem, e.g. ``scannerA`` for
    ``scannerA_0042.jpg``; ``"none"`` when the stem has no letters."""
    if not name:
        return "none"
    stem = name.rsplit(".", 1)[0].lower()
    tokens = [t for t in _TOKEN_SPLIT.split(stem) if len(t) >= 2]
    return tokens[0] if tokens else "none"


def _size_bucket(width: int, height: int) -> str:
    longest = max(width, height)
    for limit in (64, 128, 256, 512, 1024):
        if longest <= limit:
            return f"<={limit}"
    return ">1024"


def _bytes_bucket(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "unknown"
    if n_bytes <= 0:
        return "0"
    return f"2^{int(math.floor(math.log2(n_bytes)))}"


def metadata_features(sample) -> dict[str, str]:
    """All metadata features carried by one sample, as strings."""
    width, height = sample.image.size
    features = {
        "format": (sample.source_format or "unknown").upper(),
        "filename_prefix": filename_prefix(sample.name),
        "filename_has_digits": (
            "yes" if sample.name and any(c.isdigit() for c in sample.name) else "no"
        ),
        "aspect": (
            "square" if abs(width - height) <= max(2, 0.05 * max(width, height))
            else ("portrait" if height > width else "landscape")
        ),
        "size_bucket": _size_bucket(width, height),
        "file_bytes_bucket": _bytes_bucket(
            len(sample.raw_bytes) if sample.raw_bytes is not None else None
        ),
        "has_exif": "yes" if sample.exif else "no",
    }
    for key in ("exif_make", "exif_model", "exif_software"):
        features[key] = sample.exif.get(key, "absent")
    return features


def normalized_mutual_information(values: list[str], labels: list[str]) -> float:
    """MI(feature; label) / H(label): the fraction of label uncertainty a
    single categorical feature explains. 0 = independent, 1 = the feature
    fully determines the label."""
    n = len(labels)
    if n == 0:
        return 0.0
    label_counts: dict[str, int] = {}
    value_counts: dict[str, int] = {}
    joint: dict[tuple[str, str], int] = {}
    for value, label in zip(values, labels):
        label_counts[label] = label_counts.get(label, 0) + 1
        value_counts[value] = value_counts.get(value, 0) + 1
        joint[(value, label)] = joint.get((value, label), 0) + 1
    h_label = -sum((c / n) * math.log(c / n) for c in label_counts.values())
    if h_label <= 0:
        return 0.0
    mi = 0.0
    for (value, label), count in joint.items():
        p_joint = count / n
        mi += p_joint * math.log(
            p_joint / ((value_counts[value] / n) * (label_counts[label] / n))
        )
    return float(mi / h_label)


def run_metadata_check(samples, *, caution_lift: float, fail_lift: float) -> dict:
    """Run the metadata check over pooled samples (all splits combined:
    a metadata shortcut is a property of the dataset, not of one split).

    Returns a dict with ``status``, ``headline``, and JSON-safe ``details``.
    """
    labels = [s.label for s in samples]
    rows = [metadata_features(s) for s in samples]

    # Drop features that never vary: they carry no signal by construction.
    varying = [
        name
        for name in rows[0]
        if len({row[name] for row in rows}) > 1
    ] if rows else []
    if not varying:
        return {
            "status": "skipped",
            "headline": "no varying metadata found (filenames, EXIF, dimensions all uniform)",
            "details": {"features_checked": [], "reason": "no varying metadata"},
        }

    per_feature_nmi = {}
    for name in varying:
        per_feature_nmi[name] = round(
            normalized_mutual_information([row[name] for row in rows], labels), 4
        )
    top_feature = max(per_feature_nmi, key=per_feature_nmi.get)
    top_nmi = per_feature_nmi[top_feature]

    probe_rows = [{name: row[name] for name in varying} for row in rows]
    accuracy, classes_used = categorical_naive_bayes_cv_accuracy(probe_rows, labels)
    chance = majority_rate(labels)
    lift = accuracy - chance

    status = lift_status(lift, caution_lift, fail_lift)
    # A single feature that nearly determines the label is a leak on its own,
    # even if the smoothed probe dilutes it.
    if top_nmi >= 0.8:
        status = "fail"
    elif top_nmi >= 0.5 and status == "pass":
        status = "caution"

    chance_pct = f"{chance:.0%}"
    acc_pct = f"{accuracy:.0%}"
    if status == "pass":
        headline = (
            f"metadata alone predicts the label at {acc_pct} (chance {chance_pct}); "
            f"strongest single feature '{top_feature}' explains {top_nmi:.0%} of the label"
        )
    else:
        headline = (
            f"metadata alone predicts the label at {acc_pct} (chance {chance_pct}); "
            f"strongest single feature '{top_feature}' explains {top_nmi:.0%} of the label"
        )
    return {
        "status": status,
        "headline": headline,
        "details": {
            "probe": "categorical naive Bayes, 3-fold cross-validated",
            "probe_accuracy": round(float(accuracy), 4),
            "chance_rate": round(float(chance), 4),
            "lift": round(float(lift), 4),
            "classes_used": classes_used,
            "features_checked": varying,
            "per_feature_nmi": per_feature_nmi,
            "top_feature": top_feature,
            "top_feature_nmi": top_nmi,
            "n_samples": len(samples),
        },
    }
