"""Loading image splits into a uniform Sample record.

A split can be given as:

* a folder laid out as ``split/<class_name>/<image files>`` (labels come
  from the subfolder names), or
* a flat folder plus a parallel ``labels`` sequence, or
* a sequence of PIL images / NumPy arrays plus a parallel ``labels``
  sequence, with optional parallel ``names`` for the metadata check.

Everything is decoded once, converted to RGB, and (for files) the raw bytes
are kept for exact-duplicate hashing.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

# A few EXIF tags worth treating as potential shortcut carriers.
_EXIF_TAGS = {
    0x010F: "exif_make",
    0x0110: "exif_model",
    0x0131: "exif_software",
    0x0132: "exif_datetime",
}


@dataclass
class Sample:
    """One loaded image with everything the checks need."""

    image: Image.Image
    label: str
    name: str | None = None
    raw_bytes: bytes | None = None
    exif: dict[str, str] = field(default_factory=dict)
    source_format: str | None = None

    @property
    def display_name(self) -> str:
        return self.name if self.name is not None else "<array>"


def _extract_exif(image: Image.Image) -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        exif = image.getexif()
    except Exception:
        return found
    if not exif:
        return found
    for tag, key in _EXIF_TAGS.items():
        value = exif.get(tag)
        if value is not None and str(value).strip():
            found[key] = str(value).strip()
    return found


def _sample_from_bytes(data: bytes, label: str, name: str | None) -> Sample:
    image = Image.open(io.BytesIO(data))
    exif = _extract_exif(image)
    source_format = image.format
    rgb = image.convert("RGB")
    rgb.load()
    return Sample(
        image=rgb,
        label=label,
        name=name,
        raw_bytes=data,
        exif=exif,
        source_format=source_format,
    )


def _sample_from_image(image: Image.Image, label: str, name: str | None) -> Sample:
    exif = _extract_exif(image)
    rgb = image.convert("RGB")
    rgb.load()
    return Sample(image=rgb, label=label, name=name, exif=exif, source_format=image.format)


def _sample_from_array(array: np.ndarray, label: str, name: str | None) -> Sample:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.0 and float(arr.min()) >= 0.0:
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        image = Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        image = Image.fromarray(arr[:, :, :3]).convert("RGB")
    else:
        raise ValueError(f"unsupported array shape {arr.shape}; expected HxW or HxWx3/4")
    return Sample(image=image, label=label, name=name)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_split(
    source: Union[str, Path, Sequence[object]],
    labels: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
    *,
    split_name: str = "split",
) -> list[Sample]:
    """Load one split into a list of :class:`Sample`.

    ``source`` is either a folder (class subfolders, or flat with
    ``labels``) or a sequence of PIL images / NumPy arrays with ``labels``.
    """
    if isinstance(source, (str, Path)):
        root = Path(source)
        if not root.is_dir():
            raise ValueError(f"{split_name}: folder not found: {root}")
        class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        samples: list[Sample] = []
        if class_dirs:
            if labels is not None:
                raise ValueError(
                    f"{split_name}: labels were given but {root} has class "
                    "subfolders; labels come from the subfolder names"
                )
            for class_dir in class_dirs:
                for path in sorted(class_dir.iterdir()):
                    if _is_image_file(path):
                        samples.append(
                            _sample_from_bytes(path.read_bytes(), class_dir.name, path.name)
                        )
        else:
            files = sorted(p for p in root.iterdir() if _is_image_file(p))
            if labels is None:
                raise ValueError(
                    f"{split_name}: {root} is a flat folder; pass labels=... "
                    "parallel to the sorted file order"
                )
            if len(labels) != len(files):
                raise ValueError(
                    f"{split_name}: got {len(labels)} labels for {len(files)} image files"
                )
            for path, label in zip(files, labels):
                samples.append(
                    _sample_from_bytes(path.read_bytes(), str(label), path.name)
                )
        return samples

    items = list(source)
    if not items:
        return []
    if labels is None:
        raise ValueError(f"{split_name}: labels=... is required for image/array input")
    if len(labels) != len(items):
        raise ValueError(
            f"{split_name}: got {len(labels)} labels for {len(items)} images"
        )
    if names is not None and len(names) != len(items):
        raise ValueError(
            f"{split_name}: got {len(names)} names for {len(items)} images"
        )
    samples = []
    for i, item in enumerate(items):
        name = None if names is None else str(names[i])
        label = str(labels[i])
        if isinstance(item, Image.Image):
            samples.append(_sample_from_image(item, label, name))
        elif isinstance(item, np.ndarray):
            samples.append(_sample_from_array(item, label, name))
        else:
            raise ValueError(
                f"{split_name}: item {i} is {type(item).__name__}; expected a "
                "PIL image or NumPy array"
            )
    return samples
