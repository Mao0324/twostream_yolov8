#!/usr/bin/env python3
"""Create two-stream OBB weights from single-stream YOLOv8s-OBB weights.

Default paths match the user's requested locations:
  source: /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb.pt
  output: /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream.pt

Usage:
  python tools/make_twostream_obb_weights.py \
      --target-yaml /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/yaml/yolov8_twostream_obb_mbnet.yaml \
      --output /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream_mbnet.pt
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
DEFAULT_TARGET_YAML = Path("/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/yaml/yolov8_twostream_obb_mbnet.yaml")
DEFAULT_OUTPUT = Path("/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream_mbnet.pt")

# single-stream yolov8s(-obb) layer index -> two-stream RGB branch layer index
SINGLE_TO_TWOSTREAM_RGB = {
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
    22: 38,
}

# two-stream RGB branch layer index -> two-stream IR branch layer index
TWOSTREAM_RGB_TO_IR = {
    0: 4,
    1: 5,
    2: 6,
    3: 7,
    8: 9,
    11: 12,
    13: 14,
    16: 17,
    18: 19,
    20: 21,
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


def _copy_prefix(src_sd: dict, dst_sd: dict, src_prefix: str, dst_prefixes: list[str], tag: str) -> int:
    """Copy one state-dict prefix to one or more destination prefixes when shapes match."""
    copied = 0
    for key, value in src_sd.items():
        if not key.startswith(src_prefix):
            continue
        suffix = key[len(src_prefix) :]
        for dst_prefix in dst_prefixes:
            dst_key = dst_prefix + suffix
            if dst_key in dst_sd and dst_sd[dst_key].shape == value.shape:
                dst_sd[dst_key] = value.clone()
                copied += 1
    print(f"[{tag}] copied={copied}")
    return copied


def _copy_c2f_to_dual_dmaf(src_sd: dict, dst_sd: dict, src_idx: int, dst_idx: int) -> int:
    """Map a standard C2f into both branches of a DualC2fDMAF layer."""
    src_prefix = f"model.{src_idx}."
    dst_prefix = f"model.{dst_idx}."
    copied = 0
    for key, value in src_sd.items():
        if not key.startswith(src_prefix):
            continue
        suffix = key[len(src_prefix) :]
        destinations = []
        if suffix.startswith("cv1."):
            tail = suffix[len("cv1.") :]
            destinations = [f"cv1_rgb.{tail}", f"cv1_ir.{tail}"]
        elif suffix.startswith("cv2."):
            tail = suffix[len("cv2.") :]
            destinations = [f"cv2_rgb.{tail}", f"cv2_ir.{tail}"]
        elif suffix.startswith("m."):
            parts = suffix.split(".", 3)
            if len(parts) == 4 and parts[2] in {"cv1", "cv2"}:
                block_index, conv_name, tail = parts[1], parts[2], parts[3]
                destinations = [
                    f"blocks.{block_index}.rgb_{conv_name}.{tail}",
                    f"blocks.{block_index}.ir_{conv_name}.{tail}",
                ]
        for destination in destinations:
            dst_key = dst_prefix + destination
            if dst_key in dst_sd and dst_sd[dst_key].shape == value.shape:
                dst_sd[dst_key] = value.clone()
                copied += 1
    print(f"[C2f {src_idx}->DualC2fDMAF {dst_idx}] copied={copied}")
    return copied


def _initialize_mbnet_target(src_sd: dict, dst_sd: dict) -> int:
    """Initialize dense-DMAF dual backbones, dual necks, and the final OBB predictor."""
    copied = 0

    # Symmetric RGB/IR stems and downsampling layers.
    for src_idx, rgb_idx, ir_idx in ((0, 0, 2), (1, 1, 3), (3, 5, 6), (5, 8, 9), (7, 11, 12)):
        copied += _copy_prefix(
            src_sd,
            dst_sd,
            f"model.{src_idx}.",
            [f"model.{rgb_idx}.", f"model.{ir_idx}."],
            f"stem/downsample {src_idx}->{rgb_idx},{ir_idx}",
        )

    # Standard C2f parameters seed both modality-specific residual functions; DMAF itself is parameter-free.
    for src_idx, dst_idx in ((2, 4), (4, 7), (6, 10), (8, 13)):
        copied += _copy_c2f_to_dual_dmaf(src_sd, dst_sd, src_idx, dst_idx)

    copied += _copy_prefix(
        src_sd, dst_sd, "model.9.", ["model.14.rgb.", "model.14.ir."], "SPPF 9->DualSPPF 14"
    )

    # Seed both independent FPN/PAN paths from the original YOLOv8 neck.
    neck_map = {
        12: "p4",
        15: "p3",
        16: "down4",
        18: "out4",
        19: "down5",
        21: "out5",
    }
    for src_idx, name in neck_map.items():
        copied += _copy_prefix(
            src_sd,
            dst_sd,
            f"model.{src_idx}.",
            [f"model.15.rgb.{name}.", f"model.15.ir.{name}."],
            f"neck {src_idx}->{name}",
        )

    # The final IAFAOBB predictor has the same prediction submodules as the standard OBB head.
    copied += _copy_prefix(src_sd, dst_sd, "model.22.", ["model.16."], "OBB 22->IAFAOBB 16")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Build two-stream OBB checkpoint from single-stream OBB checkpoint.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Single-stream .pt checkpoint path.")
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML, help="Two-stream model YAML path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output two-stream .pt path.")
    parser.add_argument(
        "--architecture",
        choices=("auto", "legacy", "mbnet"),
        default="auto",
        help="Weight mapping layout. 'auto' detects the dense MBNet YAML by module name.",
    )
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

    architecture = args.architecture
    if architecture == "auto":
        architecture = "mbnet" if "DualC2fDMAF" in yaml_text and "IAFAOBB" in yaml_text else "legacy"
    print(f"[INFO] mapping architecture: {architecture}")

    if architecture == "mbnet":
        copied = _initialize_mbnet_target(src_sd, dst_sd)
        print(f"[mbnet total] copied={copied}")
    else:
        # Legacy ASSA/RIFusion layouts retained for reproducibility of existing experiments.
        _copy_with_layer_map(src_sd, dst_sd, SINGLE_TO_TWOSTREAM_RGB, "single->twostream_rgb")
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
