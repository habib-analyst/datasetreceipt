# Changelog

## 0.1.0 - 2026-10-28

- Initial release of DatasetReceipt.
- One-call `scan_dataset` / `scan_arrays` API over class-folder layouts, flat folders with labels, PIL images, and NumPy arrays.
- Leakage check: SHA-256 exact duplicates and NumPy DCT pHash near-duplicates (banded pair search) across the train/test split, with within-split duplicates reported separately.
- Metadata shortcut check: per-feature normalized mutual information (filename, format, dimensions, file size, EXIF Make/Model/Software) plus a cross-validated categorical naive Bayes probe.
- Background check: cross-validated nearest-centroid probe on border-ring statistics only; the image center is never used.
- `Receipt` result object with per-check `CheckResult`s, an overall go / caution / no-go verdict, a printable receipt, and JSON-safe `to_dict()` output.
