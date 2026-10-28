"""DatasetReceipt: a go / caution / no-go receipt for image datasets."""

from .core import (
    CheckResult,
    Receipt,
    ReceiptScanner,
    scan_arrays,
    scan_dataset,
)
from .hashing import find_near_duplicate_pairs, hamming, perceptual_hash

__version__ = "0.1.0"

__all__ = [
    "CheckResult",
    "Receipt",
    "ReceiptScanner",
    "find_near_duplicate_pairs",
    "hamming",
    "perceptual_hash",
    "scan_arrays",
    "scan_dataset",
]
