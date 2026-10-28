"""DatasetReceipt: a pre-training go / caution / no-go receipt for an image
dataset.

One call scans a train (and optional test) split for the three ways an
image dataset most often lies to you before training:

1. **leakage** — exact or near duplicates crossing the train/test split;
2. **metadata shortcuts** — the label predictable from filenames, EXIF,
   dimensions, or file size alone;
3. **background signal** — the label predictable from the image border
   alone, with the subject removed.

No model is trained on your data and no embeddings are needed. The only
dependencies are NumPy and Pillow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

from .background import run_background_check
from .hashing import (
    file_hash,
    find_near_duplicate_pairs,
    perceptual_hash,
    pixels_hash,
)
from .loading import Sample, load_split
from .metadata import run_metadata_check

CHECK_ORDER = ("leakage", "metadata", "background")
_STATUS_RANK = {"pass": 0, "skipped": 0, "caution": 1, "fail": 2}
_STATUS_LABEL = {"pass": "PASS", "caution": "CAUTION", "fail": "NO-GO", "skipped": "SKIPPED"}
_MAX_PAIRS_IN_DETAILS = 50


@dataclass
class CheckResult:
    """One line of the receipt."""

    name: str
    status: str  # "pass" | "caution" | "fail" | "skipped"
    headline: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "headline": self.headline,
            "details": self.details,
        }


@dataclass
class Receipt:
    """The full scan result: per-check verdicts plus one overall verdict."""

    verdict: str  # "go" | "caution" | "no-go"
    checks: dict[str, CheckResult]
    n_train: int
    n_test: int
    classes: list[str]
    config: dict = field(default_factory=dict)

    @property
    def is_go(self) -> bool:
        return self.verdict == "go"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "classes": self.classes,
            "config": self.config,
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def summary(self) -> str:
        """The human-readable receipt."""
        width = 68
        lines = ["=" * width, "DATASET RECEIPT".center(width), "=" * width]
        class_text = ", ".join(self.classes)
        lines.append(f"train   {self.n_train} images · {len(self.classes)} classes ({class_text})")
        if self.n_test:
            lines.append(f"test    {self.n_test} images")
        else:
            lines.append("test    (not provided)")
        lines.append("-" * width)
        for name in CHECK_ORDER:
            check = self.checks[name]
            label = _STATUS_LABEL[check.status]
            lines.append(f"{name:<11} {label:<8} {check.headline}")
        lines.append("-" * width)
        verdict_line = {
            "go": "OVERALL   GO — no leakage or shortcut signal found",
            "caution": "OVERALL   CAUTION — review the flagged checks before training",
            "no-go": "OVERALL   NO-GO — fix the flagged checks before training",
        }[self.verdict]
        lines.append(verdict_line)
        lines.append("=" * width)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _exact_digest(sample: Sample) -> str:
    if sample.raw_bytes is not None:
        return file_hash(sample.raw_bytes)
    return pixels_hash(sample.image)


def _pair_record(split_a: str, a: Sample, split_b: str, b: Sample, extra: dict) -> dict:
    record = {
        "a": f"{split_a}:{a.display_name}",
        "b": f"{split_b}:{b.display_name}",
        "a_label": a.label,
        "b_label": b.label,
        "same_label": a.label == b.label,
    }
    record.update(extra)
    return record


def run_leakage_check(
    train: list[Sample], test: list[Sample], *, max_distance: int
) -> CheckResult:
    """Exact and near duplicates within and across the splits."""
    if not test:
        return CheckResult(
            name="leakage",
            status="skipped",
            headline="no test split provided, so cross-split leakage was not checked",
            details={"reason": "no test split"},
        )

    train_digests = [_exact_digest(s) for s in train]
    test_digests = [_exact_digest(s) for s in test]
    train_hashes = [perceptual_hash(s.image) for s in train]
    test_hashes = [perceptual_hash(s.image) for s in test]

    # Exact duplicates across the split.
    digest_to_train: dict[str, list[int]] = {}
    for i, digest in enumerate(train_digests):
        digest_to_train.setdefault(digest, []).append(i)
    exact_pairs: list[dict] = []
    exact_test_indices: set[int] = set()
    for j, digest in enumerate(test_digests):
        for i in digest_to_train.get(digest, []):
            exact_pairs.append(
                _pair_record("train", train[i], "test", test[j], {"kind": "exact"})
            )
            exact_test_indices.add(j)

    # Near duplicates across the split (excluding pairs already exact).
    combined = train_hashes + test_hashes
    cross_near: list[dict] = []
    near_test_indices: set[int] = set()
    for i, j, distance in find_near_duplicate_pairs(combined, max_distance):
        if i < len(train) <= j:  # one side in train, the other in test
            tj = j - len(train)
            if train_digests[i] == test_digests[tj]:
                continue  # already counted as exact
            cross_near.append(
                _pair_record(
                    "train", train[i], "test", test[tj],
                    {"kind": "near", "hamming_distance": distance},
                )
            )
            near_test_indices.add(tj)

    # Duplicates inside the training split: wasteful, reported, but they do
    # not inflate a held-out score, so they never drive the verdict.
    within_exact = len(train_digests) - len(set(train_digests))
    within_near = len(find_near_duplicate_pairs(train_hashes, max_distance))

    n_test = len(test)
    compromised = len(exact_test_indices | near_test_indices)
    compromised_fraction = compromised / n_test if n_test else 0.0

    if exact_pairs:
        status = "fail"
        headline = (
            f"{len(exact_pairs)} exact duplicate pair(s) cross the train/test split "
            f"({compromised} of {n_test} test images compromised)"
        )
    elif compromised_fraction >= 0.05:
        status = "fail"
        headline = (
            f"{len(cross_near)} near-duplicate pair(s) cross the split; "
            f"{compromised_fraction:.0%} of the test split has a near-twin in train"
        )
    elif cross_near:
        status = "caution"
        headline = (
            f"{len(cross_near)} near-duplicate pair(s) cross the train/test split "
            f"({compromised} of {n_test} test images)"
        )
    else:
        status = "pass"
        headline = "no exact or near duplicates cross the train/test split"

    return CheckResult(
        name="leakage",
        status=status,
        headline=headline,
        details={
            "near_duplicate_max_hamming": max_distance,
            "exact_cross_pairs": len(exact_pairs),
            "near_cross_pairs": len(cross_near),
            "test_images_compromised": compromised,
            "test_compromised_fraction": round(compromised_fraction, 4),
            "within_train_exact_duplicates": within_exact,
            "within_train_near_pairs": within_near,
            "pairs": (exact_pairs + cross_near)[:_MAX_PAIRS_IN_DETAILS],
        },
    )


def _overall_verdict(checks: dict[str, CheckResult]) -> str:
    worst = max(_STATUS_RANK[check.status] for check in checks.values())
    return {0: "go", 1: "caution", 2: "no-go"}[worst]


class ReceiptScanner:
    """A configured scanner; use :func:`scan_dataset` for the one-call form."""

    def __init__(
        self,
        *,
        near_duplicate_threshold: int = 8,
        border_fraction: float = 0.15,
        caution_lift: float = 0.10,
        fail_lift: float = 0.25,
    ):
        if near_duplicate_threshold < 0:
            raise ValueError("near_duplicate_threshold must be >= 0")
        if not 0.0 < border_fraction < 0.5:
            raise ValueError("border_fraction must be between 0 and 0.5")
        if not 0.0 <= caution_lift < fail_lift:
            raise ValueError("expected 0 <= caution_lift < fail_lift")
        self.near_duplicate_threshold = near_duplicate_threshold
        self.border_fraction = border_fraction
        self.caution_lift = caution_lift
        self.fail_lift = fail_lift

    def scan(
        self,
        train: Union[str, Path, Sequence[object]],
        test: Union[str, Path, Sequence[object], None] = None,
        *,
        train_labels: Sequence[str] | None = None,
        test_labels: Sequence[str] | None = None,
        train_names: Sequence[str] | None = None,
        test_names: Sequence[str] | None = None,
    ) -> Receipt:
        train_samples = load_split(train, train_labels, train_names, split_name="train")
        test_samples = (
            load_split(test, test_labels, test_names, split_name="test")
            if test is not None
            else []
        )
        if not train_samples:
            raise ValueError("train split is empty")
        classes = sorted({s.label for s in train_samples})
        pooled = train_samples + test_samples

        checks: dict[str, CheckResult] = {}
        checks["leakage"] = run_leakage_check(
            train_samples, test_samples, max_distance=self.near_duplicate_threshold
        )
        if len(classes) >= 2:
            meta = run_metadata_check(
                pooled, caution_lift=self.caution_lift, fail_lift=self.fail_lift
            )
            checks["metadata"] = CheckResult("metadata", meta["status"], meta["headline"], meta["details"])
            back = run_background_check(
                pooled,
                border_fraction=self.border_fraction,
                caution_lift=self.caution_lift,
                fail_lift=self.fail_lift,
            )
            checks["background"] = CheckResult(
                "background", back["status"], back["headline"], back["details"]
            )
        else:
            single = "only one class is present, so label-prediction probes do not apply"
            checks["metadata"] = CheckResult("metadata", "skipped", single, {"reason": single})
            checks["background"] = CheckResult("background", "skipped", single, {"reason": single})

        return Receipt(
            verdict=_overall_verdict(checks),
            checks=checks,
            n_train=len(train_samples),
            n_test=len(test_samples),
            classes=classes,
            config={
                "near_duplicate_threshold": self.near_duplicate_threshold,
                "border_fraction": self.border_fraction,
                "caution_lift": self.caution_lift,
                "fail_lift": self.fail_lift,
            },
        )


def scan_dataset(
    train: Union[str, Path, Sequence[object]],
    test: Union[str, Path, Sequence[object], None] = None,
    *,
    train_labels: Sequence[str] | None = None,
    test_labels: Sequence[str] | None = None,
    train_names: Sequence[str] | None = None,
    test_names: Sequence[str] | None = None,
    **scanner_options,
) -> Receipt:
    """Scan a dataset and return its :class:`Receipt`.

    ``train`` / ``test`` are folders laid out as ``split/<class>/<images>``,
    flat folders with ``train_labels`` / ``test_labels``, or sequences of
    PIL images / NumPy arrays with labels (and optional ``*_names``).
    Scanner options: ``near_duplicate_threshold``, ``border_fraction``,
    ``caution_lift``, ``fail_lift``.
    """
    scanner = ReceiptScanner(**scanner_options)
    return scanner.scan(
        train,
        test,
        train_labels=train_labels,
        test_labels=test_labels,
        train_names=train_names,
        test_names=test_names,
    )


def scan_arrays(
    train_images: Sequence[object],
    train_labels: Sequence[str],
    test_images: Sequence[object] | None = None,
    test_labels: Sequence[str] | None = None,
    *,
    train_names: Sequence[str] | None = None,
    test_names: Sequence[str] | None = None,
    **scanner_options,
) -> Receipt:
    """Convenience wrapper of :func:`scan_dataset` for in-memory arrays."""
    return scan_dataset(
        train_images,
        test_images,
        train_labels=train_labels,
        test_labels=test_labels,
        train_names=train_names,
        test_names=test_names,
        **scanner_options,
    )
