"""Background-only label-signal check.

The question: *can the label be predicted from the border of the image
alone, with the subject removed?* If yes, the dataset carries the label in
its backgrounds — class-specific scanners, rooms, watermarks in the corner,
color casts — and a trained model will happily learn that shortcut instead
of the subject (the classic waterbird-on-water failure).

The probe is a standardized nearest-centroid classifier over border-ring
statistics, cross-validated. It never sees the center of any image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .probes import lift_status, majority_rate, nearest_centroid_cv_accuracy


def border_features(image: Image.Image, border_fraction: float = 0.15) -> np.ndarray:
    """Statistics of the border ring only: per-channel mean/std of the ring,
    plus per-channel means of the four corner blocks and four edge-mid
    blocks. The center of the image contributes nothing."""
    if not 0.0 < border_fraction < 0.5:
        raise ValueError("border_fraction must be between 0 and 0.5")
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    height, width, _ = rgb.shape
    t = max(1, int(round(min(height, width) * border_fraction)))

    ring_mask = np.ones((height, width), dtype=bool)
    ring_mask[t : height - t, t : width - t] = False
    ring = rgb[ring_mask]  # (n_ring_pixels, 3)

    features: list[float] = []
    features.extend(ring.mean(axis=0).tolist())
    features.extend(ring.std(axis=0).tolist())

    mid_h, mid_w = height // 2, width // 2
    blocks = [
        rgb[:t, :t],  # corners
        rgb[:t, width - t :],
        rgb[height - t :, :t],
        rgb[height - t :, width - t :],
        rgb[:t, mid_w - t // 2 : mid_w + (t - t // 2)],  # edge midpoints
        rgb[height - t :, mid_w - t // 2 : mid_w + (t - t // 2)],
        rgb[mid_h - t // 2 : mid_h + (t - t // 2), :t],
        rgb[mid_h - t // 2 : mid_h + (t - t // 2), width - t :],
    ]
    for block in blocks:
        if block.size:
            features.extend(block.reshape(-1, 3).mean(axis=0).tolist())
        else:
            features.extend([0.0, 0.0, 0.0])
    return np.array(features, dtype=np.float64)


def run_background_check(
    samples, *, border_fraction: float, caution_lift: float, fail_lift: float
) -> dict:
    """Run the background-only probe over pooled samples (all splits
    combined: background-label correlation is a dataset property)."""
    labels = [s.label for s in samples]
    features = np.stack([border_features(s.image, border_fraction) for s in samples])
    accuracy, classes_used = nearest_centroid_cv_accuracy(features, labels)
    if not classes_used:
        return {
            "status": "skipped",
            "headline": "not enough samples per class to run the background probe",
            "details": {"reason": "every class needs at least 3 samples"},
        }
    chance = majority_rate(labels)
    lift = accuracy - chance
    status = lift_status(lift, caution_lift, fail_lift)
    headline = (
        f"the image border alone predicts the label at {accuracy:.0%} "
        f"(chance {chance:.0%}); the center of the image was never used"
    )
    return {
        "status": status,
        "headline": headline,
        "details": {
            "probe": "nearest-centroid on border-ring statistics, 3-fold cross-validated",
            "probe_accuracy": round(float(accuracy), 4),
            "chance_rate": round(float(chance), 4),
            "lift": round(float(lift), 4),
            "classes_used": classes_used,
            "border_fraction": border_fraction,
            "n_samples": len(samples),
        },
    }
