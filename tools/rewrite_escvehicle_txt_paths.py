#!/usr/bin/env python3
"""Batch-rewrite ESCVehicle txt path prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OLD_PREFIX = "/home/ubuntu/MCONG/datasets/ESCVehicle"
DEFAULT_NEW_PREFIX = "/home/biiteam/Storage-4T/biiteam/MCONG/datasets/ESCVehicle"
DEFAULT_DATA_DIR = Path("/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/data")

DEFAULT_FILES = [
    "escvehicle_infrared_test.txt",
    "escvehicle_infrared_train.txt",
    "escvehicle_infrared_val.txt",
    "escvehicle_visible_test.txt",
    "escvehicle_visible_train.txt",
    "escvehicle_visible_val.txt",
]


def rewrite_file(path: Path, old_prefix: str, new_prefix: str, dry_run: bool) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    updated_lines = []
    changed = 0

    for line in lines:
        if line.startswith(old_prefix):
            updated_lines.append(new_prefix + line[len(old_prefix) :])
            changed += 1
        else:
            updated_lines.append(line)

    if changed and not dry_run:
        path.write_text("\n".join(updated_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite ESCVehicle txt file path prefixes.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing txt files.")
    parser.add_argument("--old-prefix", default=DEFAULT_OLD_PREFIX, help="Old path prefix to replace.")
    parser.add_argument("--new-prefix", default=DEFAULT_NEW_PREFIX, help="New path prefix.")
    parser.add_argument("--dry-run", action="store_true", help="Only print replacement counts.")
    args = parser.parse_args()

    total = 0
    for name in DEFAULT_FILES:
        file_path = args.data_dir / name
        if not file_path.exists():
            print(f"[MISS] {file_path}")
            continue
        changed = rewrite_file(file_path, args.old_prefix, args.new_prefix, args.dry_run)
        total += changed
        mode = "would change" if args.dry_run else "changed"
        print(f"[OK] {file_path}: {mode} {changed} lines")

    print(f"[DONE] total replaced lines: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
