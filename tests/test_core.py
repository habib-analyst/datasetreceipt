"""Tests for DatasetReceipt. All data is small synthetic PIL imagery."""

import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from datareceipt import (
    find_near_duplicate_pairs,
    hamming,
    perceptual_hash,
    scan_arrays,
    scan_dataset,
)
from datareceipt.loading import load_split

SIZE = 96
CLASSES = ["circle", "square", "triangle"]
SHAPE_COLORS = {"circle": (220, 60, 60), "square": (60, 220, 60), "triangle": (60, 60, 220)}
TINTS = {"circle": (205, 130, 95), "square": (95, 150, 205), "triangle": (120, 195, 120)}


def draw_shape(image: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    cx, cy, r = SIZE // 2, SIZE // 2, 22
    if label == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=SHAPE_COLORS[label])
    elif label == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=SHAPE_COLORS[label])
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=SHAPE_COLORS[label])
    return image


def noise_image(label: str, seed: int) -> Image.Image:
    """Shape on a per-image random background: no shortcut anywhere."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 45, (SIZE, SIZE, 3))
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    return draw_shape(image, label)


def tinted_image(label: str, seed: int) -> Image.Image:
    """Shape on a class-tinted background: a planted background shortcut."""
    rng = np.random.default_rng(seed)
    tint = np.array(TINTS[label], dtype=float)
    base = tint + rng.normal(0, 6, (SIZE, SIZE, 3))
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    return draw_shape(image, label)


def make_split(factory, per_class: int, seed0: int):
    images, labels = [], []
    seed = seed0
    for label in CLASSES:
        for _ in range(per_class):
            images.append(factory(label, seed))
            labels.append(label)
            seed += 1
    return images, labels


def jpeg_roundtrip(image: Image.Image, quality: int = 75) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# ---------------------------------------------------------------- hashing


def test_phash_identical_and_near():
    image = noise_image("circle", 100)
    assert hamming(perceptual_hash(image), perceptual_hash(image.copy())) == 0
    assert hamming(perceptual_hash(image), perceptual_hash(jpeg_roundtrip(image))) <= 8
    other = noise_image("circle", 999)
    assert hamming(perceptual_hash(image), perceptual_hash(other)) > 8


def test_find_near_duplicate_pairs():
    hashes = [0, 0, 0b1111, (1 << 64) - 1]
    pairs = find_near_duplicate_pairs(hashes, 4)
    assert (0, 1, 0) in pairs
    assert (0, 2, 4) in pairs
    assert (1, 2, 4) in pairs
    assert all(pair[1] != 3 for pair in pairs)


# ---------------------------------------------------------------- leakage


def test_exact_duplicate_across_split_fails():
    train_images, train_labels = make_split(noise_image, 6, 0)
    test_images, test_labels = make_split(noise_image, 4, 500)
    test_images.append(train_images[0].copy())  # leaked copy
    test_labels.append(train_labels[0])
    receipt = scan_dataset(train_images, test_images,
                           train_labels=train_labels, test_labels=test_labels)
    leakage = receipt.checks["leakage"]
    assert leakage.status == "fail"
    assert leakage.details["exact_cross_pairs"] == 1
    assert receipt.verdict == "no-go"


def test_near_duplicate_across_split_is_flagged():
    train_images, train_labels = make_split(noise_image, 6, 0)
    test_images, test_labels = make_split(noise_image, 4, 500)
    test_images.append(jpeg_roundtrip(train_images[3]))  # re-encoded leak
    test_labels.append(train_labels[3])
    receipt = scan_dataset(train_images, test_images,
                           train_labels=train_labels, test_labels=test_labels)
    leakage = receipt.checks["leakage"]
    assert leakage.status in ("caution", "fail")
    assert leakage.details["near_cross_pairs"] >= 1
    assert leakage.details["pairs"][0]["kind"] in ("near", "exact")


def test_folder_layout_end_to_end(tmp_path):
    for split, per_class, seed0 in (("train", 5, 0), ("test", 3, 300)):
        for label in CLASSES:
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for k in range(per_class):
                noise_image(label, seed0 + k).save(folder / f"img_{seed0 + k:04d}.png")
    # leak: byte-identical copy of a train file inside the test split
    source = tmp_path / "train" / "circle" / "img_0000.png"
    (tmp_path / "test" / "circle" / "img_9000.png").write_bytes(source.read_bytes())

    receipt = scan_dataset(tmp_path / "train", tmp_path / "test")
    assert receipt.n_train == 15
    assert receipt.n_test == 10
    assert receipt.checks["leakage"].status == "fail"
    assert receipt.checks["leakage"].details["exact_cross_pairs"] == 1
    assert receipt.verdict == "no-go"


def test_no_test_split_skips_leakage():
    train_images, train_labels = make_split(noise_image, 10, 0)
    receipt = scan_dataset(train_images, train_labels=train_labels)
    assert receipt.checks["leakage"].status == "skipped"
    assert receipt.verdict == "go"


# ---------------------------------------------------------------- metadata


def test_filename_shortcut_is_flagged():
    train_images, train_labels = make_split(noise_image, 15, 0)
    prefixes = {"circle": "scannerA", "square": "scannerB", "triangle": "scannerC"}
    names = []
    for i, label in enumerate(train_labels):
        if i % 5 == 4:
            names.append(f"img_{i:04d}.png")  # neutral minority
        else:
            names.append(f"{prefixes[label]}_{i:04d}.png")
    receipt = scan_dataset(train_images, train_labels=train_labels, train_names=names)
    metadata = receipt.checks["metadata"]
    assert metadata.status == "fail"
    assert metadata.details["top_feature"] == "filename_prefix"
    assert metadata.details["per_feature_nmi"]["filename_prefix"] > 0.5


def test_neutral_names_pass_metadata():
    train_images, train_labels = make_split(noise_image, 15, 0)
    names = [f"img_{i:04d}.png" for i in range(len(train_labels))]
    receipt = scan_dataset(train_images, train_labels=train_labels, train_names=names)
    assert receipt.checks["metadata"].status in ("pass", "skipped")


def test_exif_make_shortcut_is_flagged(tmp_path):
    makes = {"circle": "ScannerA", "square": "ScannerB", "triangle": "ScannerC"}
    for label in CLASSES:
        folder = tmp_path / "train" / label
        folder.mkdir(parents=True)
        for k in range(9):
            image = noise_image(label, 50 + k)
            exif = Image.Exif()
            exif[0x010F] = makes[label]  # Make tag
            image.save(folder / f"img_{k:02d}.jpg", exif=exif)
    samples = load_split(tmp_path / "train")
    assert samples[0].exif.get("exif_make") in set(makes.values())

    receipt = scan_dataset(tmp_path / "train")
    metadata = receipt.checks["metadata"]
    assert "exif_make" in metadata.details["per_feature_nmi"]
    assert metadata.details["per_feature_nmi"]["exif_make"] > 0.9
    assert metadata.status == "fail"


# ---------------------------------------------------------------- background


def test_background_shortcut_is_flagged():
    train_images, train_labels = make_split(tinted_image, 15, 0)
    test_images, test_labels = make_split(tinted_image, 5, 700)
    receipt = scan_dataset(train_images, test_images,
                           train_labels=train_labels, test_labels=test_labels)
    background = receipt.checks["background"]
    assert background.status == "fail"
    assert background.details["probe_accuracy"] > 0.9
    assert receipt.verdict == "no-go"


def test_random_backgrounds_pass():
    train_images, train_labels = make_split(noise_image, 15, 0)
    test_images, test_labels = make_split(noise_image, 5, 700)
    receipt = scan_dataset(train_images, test_images,
                           train_labels=train_labels, test_labels=test_labels)
    assert receipt.checks["background"].status == "pass"
    assert receipt.checks["background"].details["lift"] < 0.10


# ---------------------------------------------------------------- whole receipt


def test_clean_dataset_is_go_and_json_serializable():
    train_images, train_labels = make_split(noise_image, 12, 0)
    test_images, test_labels = make_split(noise_image, 5, 900)
    names_train = [f"img_{i:04d}.png" for i in range(len(train_labels))]
    names_test = [f"img_{i:04d}.png" for i in range(len(test_labels))]
    receipt = scan_arrays(
        train_images, train_labels, test_images, test_labels,
        train_names=names_train, test_names=names_test,
    )
    assert receipt.verdict == "go"
    assert receipt.is_go
    json.dumps(receipt.to_dict())  # must not raise
    parsed = json.loads(receipt.to_json())
    assert parsed["verdict"] == "go"
    text = str(receipt)
    assert "DATASET RECEIPT" in text
    assert "OVERALL" in text and "GO" in text


def test_numpy_array_input():
    train_arrays = [np.asarray(img) for img in make_split(noise_image, 8, 0)[0]]
    _, train_labels = make_split(noise_image, 8, 0)
    receipt = scan_arrays(train_arrays, train_labels)
    assert receipt.n_train == len(train_arrays)
    assert receipt.checks["metadata"].status in ("pass", "skipped")


def test_validation_errors():
    with pytest.raises(ValueError):
        scan_dataset([], train_labels=[])
    images, labels = make_split(noise_image, 3, 0)
    with pytest.raises(ValueError):
        scan_dataset(images, train_labels=labels[:-1])
    with pytest.raises(ValueError):
        scan_dataset(images)  # labels required for array input
