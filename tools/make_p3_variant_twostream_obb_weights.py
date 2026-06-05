#!/usr/bin/env python3
"""Create two-stream OBB weights for the three P3 refinement/preserve variants.

This script keeps the same initialization policy as make_twostream_obb_weights.py:
single-stream YOLOv8s-OBB weights are copied into the RGB path and matching
neck/head layers, then RGB backbone weights are duplicated into the IR path.

The three new YAML variants insert layers after the P3 neck feature, so their
single-stream-to-two-stream layer maps differ from the original best YAML.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import ultralytics
from make_twostream_obb_weights import DEFAULT_SOURCE, TWOSTREAM_RGB_TO_IR, _copy_with_layer_map
from ultralytics import YOLO


DEFAULT_YAML_DIR = REPO_ROOT / "yaml"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE.parent
DEFAULT_OUTPUT_PREFIX = "yolov8s-obb_twostream"


@dataclass(frozen=True)
class VariantSpec:
    """A P3 variant YAML and its single-stream layer mapping."""

    key: str
    yaml_name: str
    output_suffix: str
    layer_map: dict[int, int]


VARIANTS = {
    "p3detect_refine": VariantSpec(
        key="p3detect_refine",
        yaml_name="yolov8_twostream_obb_assafusion_p3detect_refine_postc2f.yaml",
        output_suffix="p3detect_refine",
        layer_map={
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 8,
            5: 11,
            6: 13,
            7: 16,
            8: 18,
            9: 20,
            12: 28,
            15: 31,
            16: 32,
            18: 34,
            19: 35,
            21: 37,
            22: 39,
        },
    ),
    "p3_scaled_preserve": VariantSpec(
        key="p3_scaled_preserve",
        yaml_name="yolov8_twostream_obb_assafusion_p3_scaled_preserve_postc2f.yaml",
        output_suffix="p3_scaled_preserve",
        layer_map={
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 8,
            5: 11,
            6: 13,
            7: 16,
            8: 18,
            9: 20,
            12: 28,
            15: 31,
            16: 33,
            18: 35,
            19: 36,
            21: 38,
            22: 39,
        },
    ),
    "p3_refine_preserve": VariantSpec(
        key="p3_refine_preserve",
        yaml_name="yolov8_twostream_obb_assafusion_p3_refine_preserve_postc2f.yaml",
        output_suffix="p3_refine_preserve",
        layer_map={
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 8,
            5: 11,
            6: 13,
            7: 16,
            8: 18,
            9: 20,
            12: 28,
            15: 31,
            16: 33,
            18: 35,
            19: 36,
            21: 38,
            22: 40,
        },
    ),
}


def _save_twostream_checkpoint(single: YOLO, spec: VariantSpec, target_yaml: Path, output: Path) -> None:
    """Build one two-stream model, copy matching weights, and save a checkpoint."""
    if not target_yaml.exists():
        raise FileNotFoundError(f"target yaml not found for {spec.key}: {target_yaml}")

    print(f"\n=== {spec.key} ===")
    print(f"[INFO] target yaml: {target_yaml}")
    two = YOLO(str(target_yaml), task="obb")
    src_sd = single.model.state_dict()
    dst_sd = two.model.state_dict()

    _copy_with_layer_map(src_sd, dst_sd, spec.layer_map, "single->twostream_rgb")

    rgb_sd = {k: v for k, v in dst_sd.items()}
    _copy_with_layer_map(rgb_sd, dst_sd, TWOSTREAM_RGB_TO_IR, "twostream_rgb->ir")

    missing_keys, unexpected_keys = two.model.load_state_dict(dst_sd, strict=False)
    print(f"[load_state_dict] missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")

    ckpt = {
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "model": deepcopy(two.model).half(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(output))
    print(f"[DONE] saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build twostream OBB weights for P3 variant YAMLs.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Single-stream YOLOv8s-OBB .pt path.")
    parser.add_argument("--yaml-dir", type=Path, default=DEFAULT_YAML_DIR, help="Directory containing variant YAMLs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output .pt files.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Prefix for generated .pt filenames.")
    parser.add_argument(
        "--variant",
        choices=["all", *VARIANTS.keys()],
        default="all",
        help="Variant to generate, or all three.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected YAMLs, outputs, and layer maps only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        if not args.dry_run:
            raise FileNotFoundError(f"source checkpoint not found: {args.source}")
    if not args.yaml_dir.exists():
        raise FileNotFoundError(f"yaml dir not found: {args.yaml_dir}")

    selected = list(VARIANTS.values()) if args.variant == "all" else [VARIANTS[args.variant]]

    if args.dry_run:
        print(f"[DRY-RUN] source: {args.source}")
        for spec in selected:
            target_yaml = args.yaml_dir / spec.yaml_name
            output = args.output_dir / f"{args.output_prefix}_{spec.output_suffix}.pt"
            print(f"\n=== {spec.key} ===")
            print(f"target yaml: {target_yaml}")
            print(f"output     : {output}")
            print(f"layer map  : {spec.layer_map}")
        return 0

    print(f"[INFO] loading source: {args.source}")
    single = YOLO(str(args.source), task="obb")

    for spec in selected:
        target_yaml = args.yaml_dir / spec.yaml_name
        output = args.output_dir / f"{args.output_prefix}_{spec.output_suffix}.pt"
        _save_twostream_checkpoint(single, spec, target_yaml, output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
