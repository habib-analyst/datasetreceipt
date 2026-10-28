# DatasetReceipt

**Before you train, get the receipt: one scan of your image dataset for cross-split leakage, metadata shortcuts, and background-only label signals — and one verdict, GO, CAUTION, or NO-GO.**

Most image datasets fail in ways that accuracy hides until deployment: test images that are copies of train images, one class that came from a different scanner, labels you can read off the background. DatasetReceipt checks for those three failure families *before* training, with no model, no embeddings, and no dependencies beyond NumPy and Pillow — and instead of galleries to browse, it hands you a receipt.

> Name note: the working name was **DataReceipt**, but that name is already used by an unrelated data-download service and its R client, so the package is **DatasetReceipt** (`datasetreceipt`) — free on PyPI and GitHub, and it says what the tool does.

## Install

DatasetReceipt is not on PyPI yet. Install the GitHub version:

```bash
pip install "git+https://github.com/habib-analyst/datasetreceipt.git"
```

For development:

```bash
git clone https://github.com/habib-analyst/datasetreceipt.git
cd datasetreceipt
python -m pip install -e ".[dev]"
python -m pytest
```

## Quickstart

```python
from datareceipt import scan_dataset

receipt = scan_dataset("data/train", "data/test")  # folders: split/<class>/<images>
print(receipt)
```

```text
====================================================================
                          DATASET RECEIPT
====================================================================
train   54 images · 3 classes (circle, square, triangle)
test    21 images
--------------------------------------------------------------------
leakage     NO-GO    1 exact duplicate pair(s) cross the train/test split (3 of 21 test images compromised)
metadata    NO-GO    metadata alone predicts the label at 88% (chance 33%); strongest single feature 'filename_prefix' explains 76% of the label
background  NO-GO    the image border alone predicts the label at 100% (chance 33%); the center of the image was never used
--------------------------------------------------------------------
OVERALL   NO-GO — fix the flagged checks before training
====================================================================
```

(This receipt is real output from `examples/receipt_demo.py`, which plants exactly those three problems in synthetic data and catches all of them.)

The result is a first-class object, not just text:

| Field / method | Meaning |
|---|---|
| `receipt.verdict` | `"go"`, `"caution"`, or `"no-go"` — the worst of the check verdicts |
| `receipt.checks["leakage"]` | A `CheckResult` with `.status`, `.headline`, and `.details` |
| `receipt.checks["leakage"].details["pairs"]` | The actual duplicate pairs (file names, labels, Hamming distance) |
| `receipt.to_dict()` / `receipt.to_json()` | The whole receipt as JSON-safe data, for CI gates and logs |
| `receipt.summary()` | The printable receipt above |

Inputs are flexible: class-subfolder folders, flat folders with a `labels=` list, or in-memory PIL images / NumPy arrays via `scan_dataset(images, train_labels=...)` or `scan_arrays(...)`. Filenames can be passed with `train_names=` / `test_names=` so the metadata check can see them.

## The three checks

**Leakage.** Exact duplicates are found by SHA-256 of the file bytes (or decoded pixels for arrays); near-duplicates by a 64-bit perceptual hash (DCT pHash, NumPy-only) with a configurable Hamming threshold (default 8). Only pairs that *cross* the train/test split drive the verdict: any exact cross-split duplicate is a NO-GO, ≥ 5% of the test split with a near-twin in train is a NO-GO, and any other cross-split near-duplicate is a CAUTION. Duplicates *inside* the train split are counted in the details but never fail the receipt — they waste compute, they don't inflate a held-out score.

**Metadata shortcuts.** Everything except the pixels: filename prefix and digit pattern, format, aspect, dimension and file-size buckets, and EXIF Make / Model / Software. Each feature is scored by normalized mutual information with the label (the details name the strongest single feature and how much of the label it explains), and a cross-validated categorical naive Bayes probe over all features together reports how well metadata alone predicts the label versus the majority-class chance rate.

**Background signal.** A cross-validated nearest-centroid probe is trained on border-ring statistics only — ring and corner/edge block means and spreads. The center of the image is never used. If the border alone predicts the label well above chance, your backgrounds carry the label, and a real model will learn that shortcut.

Probe verdicts use the probe's *lift* over chance: ≥ 0.10 is CAUTION, ≥ 0.25 is NO-GO; a single metadata feature that explains ≥ 80% of the label is a NO-GO on its own. All thresholds are configurable via `scan_dataset(..., caution_lift=..., fail_lift=..., near_duplicate_threshold=..., border_fraction=...)` or a reusable `ReceiptScanner`.

## What DatasetReceipt is not

- **Not a dataset cleaner.** It finds and names problems; it does not delete, relabel, or re-split your data. Tools like fastdup and CleanVision are curation platforms with galleries for browsing and fixing issues — DatasetReceipt is the five-minute gate you run *before* that, and before training.
- **Not a label-error detector.** It never trains a model on your labels, so it cannot judge whether a label is wrong (that is cleanlab's job, and it needs model outputs or embeddings). DatasetReceipt needs nothing but the images and their split.
- **Not an embedding tool.** No CLIP, no feature vectors, no GPU. The perceptual hash catches re-encodings, mild resizes, and small edits — it is not invariant to crops, rotations, or heavy augmentation, and the receipt does not claim otherwise.
- **Not a benchmark.** The example uses synthetic data to demonstrate the mechanics. The numbers it prints are the planted problems being caught, not a measured accuracy on real datasets.

## Example

```bash
python examples/receipt_demo.py
```

Builds a "dirty" synthetic dataset (class-tinted backgrounds, class-coded filename prefixes, three train images planted into the test split) and a clean one, scans both, prints both receipts, and writes them to `examples/output/`. It exits non-zero if the planted problems are not caught.

## License

MIT. See `LICENSE`.
