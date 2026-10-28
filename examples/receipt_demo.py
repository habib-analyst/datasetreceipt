"""DatasetReceipt demo on synthetic data.

MECHANICS DEMO, NOT A BENCHMARK. This script generates two tiny synthetic
datasets with PIL:

* a "dirty" dataset with three planted problems — class-tinted backgrounds,
  class-coded filename prefixes, and copies of train images planted in the
  test split (one byte-identical, two re-encoded as JPEG);
* a "clean" dataset — same shapes, random backgrounds, neutral filenames,
  no copies.

It scans both and prints the receipts. The dirty dataset must come back
NO-GO with all three checks flagged; the clean one must come back GO.
Receipts are also written to examples/output/.
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datareceipt import scan_dataset  # noqa: E402

SIZE = 96
CLASSES = ["circle", "square", "triangle"]
SHAPE_COLORS = {"circle": (220, 60, 60), "square": (60, 220, 60), "triangle": (60, 60, 220)}
TINTS = {"circle": (205, 130, 95), "square": (95, 150, 205), "triangle": (120, 195, 120)}
PREFIXES = {"circle": "scannerA", "square": "scannerB", "triangle": "scannerC"}
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def draw_shape(image: Image.Image, label: str, cx: int = SIZE // 2,
               cy: int = SIZE // 2, r: int = 22) -> Image.Image:
    draw = ImageDraw.Draw(image)
    if label == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=SHAPE_COLORS[label])
    elif label == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=SHAPE_COLORS[label])
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=SHAPE_COLORS[label])
    return image


def dirty_image(label: str, seed: int) -> Image.Image:
    # Class-level tint (the planted background shortcut) plus per-image
    # jitter in tint, position, and size, and a per-image low-frequency
    # texture field, so distinct images are genuinely distinct scenes and
    # only the planted copies count as near-duplicates.
    rng = np.random.default_rng(seed)
    tint = np.array(TINTS[label], dtype=float) + rng.normal(0, 10, 3)
    small = Image.fromarray(rng.integers(0, 255, (12, 12), dtype=np.uint8))
    texture = np.asarray(small.resize((SIZE, SIZE), Image.BILINEAR), dtype=float) - 127.5
    base = tint + 0.35 * texture[:, :, None] + rng.normal(0, 6, (SIZE, SIZE, 3))
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    cx = SIZE // 2 + int(rng.integers(-12, 13))
    cy = SIZE // 2 + int(rng.integers(-12, 13))
    r = int(rng.integers(15, 29))
    return draw_shape(image, label, cx, cy, r)


def clean_image(label: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 45, (SIZE, SIZE, 3))
    return draw_shape(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)), label)


def build_dataset(root: Path, kind: str) -> None:
    factory = dirty_image if kind == "dirty" else clean_image
    seed = 0
    train_sources: dict[str, Path] = {}
    for split, per_class in (("train", 18), ("test", 6)):
        for label in CLASSES:
            folder = root / split / label
            folder.mkdir(parents=True)
            for k in range(per_class):
                image = factory(label, seed)
                if kind == "dirty" and k % 4 != 3:
                    name = f"{PREFIXES[label]}_{seed:04d}.png"
                else:
                    name = f"img_{seed:04d}.png"
                path = folder / name
                image.save(path)
                if split == "train" and label not in train_sources:
                    train_sources[label] = path
                seed += 1
    if kind == "dirty":
        test_root = root / "test"
        # Plant 1: a byte-identical copy of a train image in the test split.
        shutil.copyfile(
            train_sources["circle"], test_root / "circle" / "img_9001.png"
        )
        # Plant 2: two train images re-encoded as JPEG inside the test split.
        for label, out_name in (("square", "img_9002.jpg"), ("triangle", "img_9003.jpg")):
            buffer = io.BytesIO()
            Image.open(train_sources[label]).save(buffer, format="JPEG", quality=75)
            (test_root / label / out_name).write_bytes(buffer.getvalue())


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for kind, expected in (("dirty", "no-go"), ("clean", "go")):
            root = Path(tmp) / kind
            build_dataset(root, kind)
            receipt = scan_dataset(root / "train", root / "test")
            print(f"\n##### {kind} dataset #####")
            print(receipt)
            (OUTPUT_DIR / f"{kind}_receipt.txt").write_text(str(receipt) + "\n")
            (OUTPUT_DIR / f"{kind}_receipt.json").write_text(receipt.to_json() + "\n")
            if receipt.verdict != expected:
                failures.append(f"{kind}: expected verdict {expected!r}, got {receipt.verdict!r}")
            if kind == "dirty":
                for check in ("leakage", "metadata", "background"):
                    if receipt.checks[check].status != "fail":
                        failures.append(
                            f"dirty: expected {check} to fail, got {receipt.checks[check].status}"
                        )
    if failures:
        print("\nDEMO FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nDemo OK: the planted problems were caught, the clean dataset passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
