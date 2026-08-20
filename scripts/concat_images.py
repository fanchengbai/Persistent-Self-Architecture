# -*- coding: utf-8 -*-
"""Horizontally concatenate matching images from two directories.

Images are matched by filename stem (for example, ``a.jpg`` matches
``a.png``).  The resulting image keeps each input's original size; shorter
images are vertically centred on a configurable background.

Requires: Pillow (``pip install Pillow``)

Unicode directory and filename support is built in.  For example, this works
on Windows: ``python concat_images.py "D:\\图片\\左侧" "D:\\图片\\右侧" "D:\\输出"``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def images_by_stem(directory: Path) -> dict[str, Path]:
    """Return supported files directly inside *directory*, indexed by stem."""
    files: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.stem.casefold()
        if key in files:
            raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
        files[key] = path
    return files


def normalize_directory(path: Path) -> Path:
    """Expand a supplied path without converting it through a byte encoding.

    ``pathlib`` passes Unicode paths directly to Windows APIs, so Chinese (and
    other non-ASCII) directory names remain intact.
    """
    return path.expanduser().resolve()


def concatenate(left_path: Path, right_path: Path, output_path: Path, background: str) -> None:
    """Place two images side by side, vertically centred, and save as PNG."""
    with Image.open(left_path) as left_source, Image.open(right_path) as right_source:
        left = left_source.convert("RGBA")
        right = right_source.convert("RGBA")
        height = max(left.height, right.height)
        result = Image.new("RGBA", (left.width + right.width, height), background)
        result.alpha_composite(left, (0, (height - left.height) // 2))
        result.alpha_composite(right, (left.width, (height - right.height) // 2))
        result.save(output_path, "PNG")


def main() -> int:
    parser = argparse.ArgumentParser(description="Horizontally concatenate images with matching filenames.")
    parser.add_argument("left_dir", type=Path, help="First image directory (left side).")
    parser.add_argument("right_dir", type=Path, help="Second image directory (right side).")
    parser.add_argument("output_dir", type=Path, help="Directory for combined PNG images.")
    parser.add_argument("--background", default="white", help="Pillow colour for empty padding (default: white).")
    parser.add_argument("--strict", action="store_true", help="Fail if either directory has unmatched files.")
    args = parser.parse_args()

    args.left_dir = normalize_directory(args.left_dir)
    args.right_dir = normalize_directory(args.right_dir)
    args.output_dir = args.output_dir.expanduser().resolve()

    # Make status messages readable in older Windows terminals too.  This does
    # not alter paths; pathlib already retains them as Unicode strings.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    for directory in (args.left_dir, args.right_dir):
        if not directory.is_dir():
            parser.error(f"Not a directory: {directory}")

    try:
        left_images = images_by_stem(args.left_dir)
        right_images = images_by_stem(args.right_dir)
    except ValueError as exc:
        parser.error(str(exc))

    shared = sorted(left_images.keys() & right_images.keys())
    left_only = sorted(left_images.keys() - right_images.keys())
    right_only = sorted(right_images.keys() - left_images.keys())
    if args.strict and (left_only or right_only):
        parser.error(f"Unmatched files — left only: {left_only}; right only: {right_only}")
    if not shared:
        parser.error("No matching image filenames were found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for stem in shared:
        try:
            concatenate(left_images[stem], right_images[stem], args.output_dir / f"{stem}.png", args.background)
            print(f"Created: {stem}.png")
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            failures += 1
            print(f"Skipped {stem}: {exc}", file=sys.stderr)

    print(f"Completed: {len(shared) - failures}/{len(shared)} images created.")
    if left_only:
        print(f"Left-only files skipped: {', '.join(left_only)}")
    if right_only:
        print(f"Right-only files skipped: {', '.join(right_only)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
