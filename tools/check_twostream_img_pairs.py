#!/usr/bin/env python3
"""Diagnose RGB/IR pairing issues for this two-stream YOLO fork."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}


def img2label_paths(img_paths: list[str]) -> list[str]:
    sa = f"{Path('/').anchor}images{Path('/').anchor}"
    sb = f"{Path('/').anchor}labels{Path('/').anchor}"
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def resolve_dataset_path(data: dict, key: str) -> Path | list[Path] | None:
    value = data.get(key)
    if not value:
        return None

    yaml_file = Path(data.get("yaml_file", "")).resolve()
    root = Path(data.get("path") or yaml_file.parent)
    if not root.is_absolute():
        root = (yaml_file.parent / root).resolve()

    def _resolve_one(item: str) -> Path:
        p = Path(item)
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    if isinstance(value, (list, tuple)):
        return [_resolve_one(str(x)) for x in value]
    return _resolve_one(str(value))


def collect_images(path: Path | list[Path]) -> tuple[list[Path], list[Path]]:
    paths = path if isinstance(path, list) else [path]
    files: list[Path] = []
    missing_paths: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(x for x in p.rglob("*") if x.is_file())
        elif p.is_file():
            parent = p.parent
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                files.append((parent / line[2:]).resolve() if line.startswith("./") else Path(line).resolve())
        else:
            missing_paths.append(p)
            print(f"[MISSING PATH] {p}")
    return sorted(x for x in files if x.suffix.lower().lstrip(".") in IMG_FORMATS), missing_paths


def report_duplicates(name: str, files: list[Path], limit: int) -> None:
    counts = Counter(p.stem for p in files)
    duplicates = [stem for stem, count in counts.items() if count > 1]
    if not duplicates:
        return

    print(f"\n{name} duplicate stems: {len(duplicates)}")
    for stem in duplicates[:limit]:
        for p in files:
            if p.stem == stem:
                print(f"  {stem}: {p}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RGB/IR image pairing for a dataset YAML.")
    parser.add_argument("--data", type=Path, default=Path("data/dronevehicle.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=50, help="Maximum missing/duplicate examples to print.")
    parser.add_argument("--no-labels", action="store_true", help="Skip label existence checks for RGB images.")
    args = parser.parse_args()

    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    data["yaml_file"] = str(args.data.resolve())
    rgb_key = args.split
    ir_key = f"{args.split}_ir"
    rgb_path = resolve_dataset_path(data, rgb_key)
    ir_path = resolve_dataset_path(data, ir_key)

    if rgb_path is None:
        raise SystemExit(f"YAML key missing: {rgb_key}")
    if ir_path is None:
        raise SystemExit(f"YAML key missing: {ir_key}")

    rgb_files, missing_rgb_paths = collect_images(rgb_path)
    ir_files, missing_ir_paths = collect_images(ir_path)
    rgb_stems = {p.stem for p in rgb_files}
    ir_stems = {p.stem for p in ir_files}
    only_rgb = sorted(rgb_stems - ir_stems)
    only_ir = sorted(ir_stems - rgb_stems)

    print(f"data: {args.data.resolve()}")
    print(f"split: {args.split}")
    print(f"RGB path: {rgb_path}")
    print(f"IR path:  {ir_path}")
    print(f"RGB images: {len(rgb_files)} ({len(rgb_stems)} unique stems)")
    print(f"IR images:  {len(ir_files)} ({len(ir_stems)} unique stems)")

    print(f"\nOnly in RGB: {len(only_rgb)}")
    for stem in only_rgb[: args.limit]:
        matches = [p for p in rgb_files if p.stem == stem]
        print(f"  {stem}: {matches[0] if matches else ''}")

    print(f"\nOnly in IR: {len(only_ir)}")
    for stem in only_ir[: args.limit]:
        matches = [p for p in ir_files if p.stem == stem]
        print(f"  {stem}: {matches[0] if matches else ''}")

    report_duplicates("RGB", rgb_files, args.limit)
    report_duplicates("IR", ir_files, args.limit)

    first_mismatch = None
    for i, (rgb, ir) in enumerate(zip(rgb_files, ir_files)):
        if rgb.stem != ir.stem:
            first_mismatch = (i, rgb, ir)
            break
    if first_mismatch:
        i, rgb, ir = first_mismatch
        print(f"\nFirst sorted-list stem mismatch at index {i}:")
        print(f"  RGB: {rgb}")
        print(f"  IR:  {ir}")

    if not args.no_labels:
        labels = [Path(x) for x in img2label_paths([str(p) for p in rgb_files])]
        missing_labels = [p for p in labels if not p.exists()]
        print(f"\nMissing labels for RGB images: {len(missing_labels)}")
        for p in missing_labels[: args.limit]:
            print(f"  {p}")

        cache_files = sorted(set(p.parent / "labels.cache" for p in labels if (p.parent / "labels.cache").exists()))
        if cache_files:
            print("\nExisting label caches; delete them after changing files:")
            for p in cache_files:
                print(f"  {p}")

    if missing_rgb_paths or missing_ir_paths or only_rgb or only_ir or len(rgb_files) != len(ir_files) or first_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
