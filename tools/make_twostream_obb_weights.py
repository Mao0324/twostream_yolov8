#!/usr/bin/env python3
"""Create two-stream OBB weights from single-stream YOLOv8s-OBB weights.

Default paths match the user's requested locations:
  source: /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb.pt
  output: /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream.pt

Usage:
  python make_twostream_obb_weights.py \
      --target-yaml /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/yaml/PC2f_MPF_yolov8s.yaml
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


DEFAULT_SOURCE = Path("/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb.pt")
DEFAULT_TARGET_YAML = Path("/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/yaml/yolov8_twostream_sparse_ri_obb.yaml")
DEFAULT_OUTPUT = Path("/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream.pt")

# single-stream yolov8s(-obb) layer index -> two-stream RGB branch layer index
SINGLE_TO_TWOSTREAM_RGB = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 9,
    5: 10,
    6: 14,
    7: 15,
    8: 19,
    9: 20,
    12: 28,
    15: 31,
    16: 32,
    18: 34,
    19: 35,
    21: 37,
    22: 38,
}

# two-stream RGB branch layer index -> two-stream IR branch layer index
TWOSTREAM_RGB_TO_IR = {
    0: 4,
    1: 5,
    2: 6,
    3: 7,
    9: 11,
    10: 12,
    14: 16,
    15: 17,
    19: 21,
    20: 22,
}


def _map_layer_key(key: str, src_idx: int, dst_idx: int) -> str | None:
    prefix = f"model.{src_idx}."
    if key.startswith(prefix):
        return f"model.{dst_idx}." + key[len(prefix) :]
    return None


def _copy_with_layer_map(src_sd: dict, dst_sd: dict, layer_map: dict[int, int], tag: str) -> tuple[int, int, int]:
    copied = 0
    miss = 0
    mismatch = 0
    for k, v in src_sd.items():
        mapped = None
        for src_idx, dst_idx in layer_map.items():
            mapped = _map_layer_key(k, src_idx, dst_idx)
            if mapped:
                break
        if not mapped:
            continue
        if mapped not in dst_sd:
            miss += 1
            continue
        if dst_sd[mapped].shape != v.shape:
            mismatch += 1
            continue
        dst_sd[mapped] = v.clone()
        copied += 1
    print(f"[{tag}] copied={copied}, missing={miss}, shape_mismatch={mismatch}")
    return copied, miss, mismatch


def main() -> int:
    parser = argparse.ArgumentParser(description="Build two-stream OBB checkpoint from single-stream OBB checkpoint.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Single-stream .pt checkpoint path.")
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML, help="Two-stream model YAML path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output two-stream .pt path.")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"source checkpoint not found: {args.source}")
    if not args.target_yaml.exists():
        raise FileNotFoundError(f"target yaml not found: {args.target_yaml}")

    yaml_text = args.target_yaml.read_text(encoding="utf-8", errors="ignore")
    if "OBB" not in yaml_text:
        print(
            "[WARN] target yaml does not contain 'OBB' head. "
            "Script can still run, but for OBB training you should use an OBB-head yaml."
        )

    print(f"[INFO] loading source: {args.source}")
    single = YOLO(str(args.source), task="obb")
    src_sd = single.model.state_dict()

    print(f"[INFO] building two-stream model from: {args.target_yaml}")
    two = YOLO(str(args.target_yaml), task="obb")
    dst_sd = two.model.state_dict()

    # Step-1: map single-stream weights to two-stream RGB path (and shared neck/head where shape matches).
    _copy_with_layer_map(src_sd, dst_sd, SINGLE_TO_TWOSTREAM_RGB, "single->twostream_rgb")

    # Step-2: duplicate RGB backbone weights to IR branch.
    # Use current dst_sd as source so we copy exactly what RGB branch already has.
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(args.output))
    print(f"[DONE] saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

