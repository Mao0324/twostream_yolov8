#!/usr/bin/env python3
"""Create two-stream OBB weights from single-stream YOLOv8s-OBB weights.

Default paths use this repository checkout:
  source: /home/ubuntu/MCONG/MCONG/twostream_yolov8/pre-trained/yolov8s-obb.pt
  output: /home/ubuntu/MCONG/MCONG/twostream_yolov8/pre-trained/yolov8s-obb_twostream_mbnet.pt

Usage:
  python tools/make_twostream_obb_weights.py \
      --target-yaml /home/ubuntu/MCONG/MCONG/twostream_yolov8/yaml/yolov8_twostream_obb_mbnet.yaml \
      --output /home/ubuntu/MCONG/MCONG/twostream_yolov8/pre-trained/yolov8s-obb_twostream_mbnet.pt
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import ultralytics
from ultralytics import YOLO


_TORCH_LOAD = torch.load


def _torch_load_trusted_checkpoint(*args, **kwargs):
    """Load trusted local YOLO checkpoints across PyTorch versions."""
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


DEFAULT_SOURCE = Path("/home/ubuntu/MCONG/MCONG/twostream_yolov8/pre-trained/yolov8s-obb.pt")
DEFAULT_TARGET_YAML = Path("/home/ubuntu/MCONG/MCONG/twostream_yolov8/yaml/yolov8_twostream_obb_assafusion_postc2f_iafa_before_fpn_lastdmaf_fasterir_p2c2f.yaml")
DEFAULT_OUTPUT = Path("/home/ubuntu/MCONG/MCONG/twostream_yolov8/pre-trained/yolov8s-obb_twostream.pt")

# single-stream yolov8s(-obb) layer index -> legacy ASSA two-stream RGB branch layer index
LEGACY_SINGLE_TO_TWOSTREAM_RGB = {
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

# legacy ASSA two-stream RGB branch layer index -> IR branch layer index
LEGACY_TWOSTREAM_RGB_TO_IR = {
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


def _copy_c2f_to_last_dmaf_fasterir(src_sd: dict, dst_sd: dict, src_idx: int, dst_idx: int) -> int:
    """Map a standard C2f into RGB-side weights of DualC2fLastDMAF_FasterIR."""
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
                block_index, conv_name, tail = int(parts[1]), parts[2], parts[3]
                if f"blocks.{block_index}.rgb_{conv_name}.{tail}" in dst_sd:
                    destinations = [f"blocks.{block_index}.rgb_{conv_name}.{tail}"]
                else:
                    destinations = [f"last_dmaf.rgb_{conv_name}.{tail}"]
        for destination in destinations:
            dst_key = dst_prefix + destination
            if dst_key in dst_sd and dst_sd[dst_key].shape == value.shape:
                dst_sd[dst_key] = value.clone()
                copied += 1
    print(f"[C2f {src_idx}->DualC2fLastDMAF_FasterIR {dst_idx}] copied={copied}")
    return copied


def _copy_standard_backbone_to_dual_dmaf(src_sd: dict, dst_sd: dict, layout: dict) -> int:
    """Initialize symmetric dense-DMAF backbones from a single-stream YOLOv8 backbone."""
    copied = 0

    # Symmetric RGB/IR stems and downsampling layers.
    for src_idx, rgb_idx, ir_idx in layout["downsample_pairs"]:
        copied += _copy_prefix(
            src_sd,
            dst_sd,
            f"model.{src_idx}.",
            [f"model.{rgb_idx}.", f"model.{ir_idx}."],
            f"stem/downsample {src_idx}->{rgb_idx},{ir_idx}",
        )

    # Standard C2f parameters seed both modality-specific residual functions; DMAF itself is parameter-free.
    for src_idx, dst_idx in layout["dmaf_layers"]:
        copied += _copy_c2f_to_dual_dmaf(src_sd, dst_sd, src_idx, dst_idx)
    return copied


def _copy_standard_backbone_to_last_dmaf(src_sd: dict, dst_sd: dict, layout: dict) -> int:
    """Initialize P2 baseline blocks plus P3-P5 last-DMAF FasterIR blocks."""
    copied = 0
    for src_idx, rgb_idx, ir_idx in layout["downsample_pairs"]:
        copied += _copy_prefix(
            src_sd,
            dst_sd,
            f"model.{src_idx}.",
            [f"model.{rgb_idx}.", f"model.{ir_idx}."],
            f"stem/downsample {src_idx}->{rgb_idx},{ir_idx}",
        )
    for src_idx, rgb_idx, ir_idx in layout["p2_blocks"]:
        copied += _copy_prefix(
            src_sd,
            dst_sd,
            f"model.{src_idx}.",
            [f"model.{rgb_idx}.", f"model.{ir_idx}."],
            f"P2 baseline C2f {src_idx}->{rgb_idx},{ir_idx}",
        )
    for src_idx, dst_idx in layout["last_dmaf_layers"]:
        copied += _copy_c2f_to_last_dmaf_fasterir(src_sd, dst_sd, src_idx, dst_idx)
    return copied


def _copy_obb_head(src_sd: dict, dst_sd: dict, dst_idx: int, tag: str) -> int:
    """Copy the standard YOLOv8-OBB prediction branches into an OBB-compatible target head."""
    copied = 0
    for submodule in ("cv2.", "cv3.", "cv4.", "dfl."):
        copied += _copy_prefix(src_sd, dst_sd, f"model.22.{submodule}", [f"model.{dst_idx}.{submodule}"], tag)
    return copied


MBNET_LAYOUT = {
    "downsample_pairs": ((0, 0, 2), (1, 1, 3), (3, 5, 6), (5, 8, 9), (7, 11, 12)),
    "dmaf_layers": ((2, 4), (4, 7), (6, 10), (8, 13)),
    "dual_sppf": 14,
    "dual_fpn": 15,
    "head": 16,
}

MBNET_EXPANDED_LAYOUT = {
    "downsample_pairs": ((0, 0, 2), (1, 1, 3), (3, 5, 6), (5, 8, 9), (7, 11, 12)),
    "dmaf_layers": ((2, 4), (4, 7), (6, 10), (8, 13)),
    "dual_sppf": 14,
    "rgb_neck": {12: 23, 15: 26, 16: 27, 18: 29, 19: 30, 21: 32},
    "ir_neck": {12: 35, 15: 38, 16: 39, 18: 41, 19: 42, 21: 44},
    "head": 45,
}

ASSA_DMAF_IAFA_LAYOUT = {
    "downsample_pairs": ((0, 0, 2), (1, 1, 3), (3, 5, 6), (5, 9, 10), (7, 13, 14)),
    "dmaf_layers": ((2, 4), (4, 7), (6, 11), (8, 15)),
    "sppf_pairs": ((9, 16, 17),),
    "rgb_neck": {12: 27, 15: 30, 16: 31, 18: 33, 19: 34, 21: 36},
    "ir_neck": {12: 39, 15: 42, 16: 43, 18: 45, 19: 46, 21: 48},
    "head": 49,
}

ASSA_LASTDMAF_FASTIR_P2C2F_LAYOUT = {
    "downsample_pairs": ((0, 0, 2), (1, 1, 3), (3, 6, 7), (5, 10, 11), (7, 14, 15)),
    "p2_blocks": ((2, 4, 5),),
    "last_dmaf_layers": ((4, 8), (6, 12), (8, 16)),
    "sppf_pairs": ((9, 17, 18),),
    "rgb_neck": {12: 28, 15: 31, 16: 32, 18: 34, 19: 35, 21: 37},
    "ir_neck": {12: 40, 15: 43, 16: 44, 18: 46, 19: 47, 21: 49},
    "head": 50,
}

ASSA_IAFA_BEFORE_FPN_LASTDMAF_FASTIR_P2C2F_LAYOUT = {
    "downsample_pairs": ((0, 0, 2), (1, 1, 3), (3, 6, 7), (5, 11, 12), (7, 16, 17)),
    "p2_blocks": ((2, 4, 5),),
    "last_dmaf_layers": ((4, 8), (6, 13), (8, 18)),
    "sppf_pairs": ((9, 19, 20),),
    "neck": {12: 25, 15: 28, 16: 29, 18: 31, 19: 32, 21: 34},
    "head": 35,
}


def _initialize_mbnet_target(src_sd: dict, dst_sd: dict, layout: dict = MBNET_LAYOUT) -> int:
    """Initialize dense-DMAF dual backbones, dual necks, and the final OBB predictor."""
    copied = _copy_standard_backbone_to_dual_dmaf(src_sd, dst_sd, layout)

    copied += _copy_prefix(
        src_sd,
        dst_sd,
        "model.9.",
        [f"model.{layout['dual_sppf']}.rgb.", f"model.{layout['dual_sppf']}.ir."],
        f"SPPF 9->DualSPPF {layout['dual_sppf']}",
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
            [f"model.{layout['dual_fpn']}.rgb.{name}.", f"model.{layout['dual_fpn']}.ir.{name}."],
            f"neck {src_idx}->{name}",
        )

    # The final IAFAOBB predictor has the same prediction submodules as the standard OBB head.
    copied += _copy_obb_head(src_sd, dst_sd, layout["head"], f"OBB 22->IAFAOBB {layout['head']}")
    return copied


def _initialize_expanded_iafa_target(src_sd: dict, dst_sd: dict, layout: dict, tag: str) -> int:
    """Initialize expanded dual-neck IAFA layouts from single-stream YOLOv8-OBB weights."""
    copied = _copy_standard_backbone_to_dual_dmaf(src_sd, dst_sd, layout)

    if "dual_sppf" in layout:
        copied += _copy_prefix(
            src_sd,
            dst_sd,
            "model.9.",
            [f"model.{layout['dual_sppf']}.rgb.", f"model.{layout['dual_sppf']}.ir."],
            f"SPPF 9->DualSPPF {layout['dual_sppf']}",
        )
    for src_idx, rgb_idx, ir_idx in layout.get("sppf_pairs", ()):
        copied += _copy_prefix(
            src_sd,
            dst_sd,
            f"model.{src_idx}.",
            [f"model.{rgb_idx}.", f"model.{ir_idx}."],
            f"SPPF {src_idx}->{rgb_idx},{ir_idx}",
        )

    for src_idx, dst_idx in layout["rgb_neck"].items():
        copied += _copy_prefix(src_sd, dst_sd, f"model.{src_idx}.", [f"model.{dst_idx}."], f"{tag} RGB neck {src_idx}->{dst_idx}")
    for src_idx, dst_idx in layout["ir_neck"].items():
        copied += _copy_prefix(src_sd, dst_sd, f"model.{src_idx}.", [f"model.{dst_idx}."], f"{tag} IR neck {src_idx}->{dst_idx}")

    copied += _copy_obb_head(src_sd, dst_sd, layout["head"], f"OBB 22->IAFAOBB {layout['head']}")
    return copied


def _initialize_last_dmaf_iafa_target(src_sd: dict, dst_sd: dict, layout: dict, tag: str) -> int:
    """Initialize P2-baseline + P3-P5 last-DMAF dual-neck IAFA layouts."""
    copied = _copy_standard_backbone_to_last_dmaf(src_sd, dst_sd, layout)

    for src_idx, rgb_idx, ir_idx in layout.get("sppf_pairs", ()):
        copied += _copy_prefix(
            src_sd,
            dst_sd,
            f"model.{src_idx}.",
            [f"model.{rgb_idx}.", f"model.{ir_idx}."],
            f"SPPF {src_idx}->{rgb_idx},{ir_idx}",
        )

    for src_idx, dst_idx in layout["rgb_neck"].items():
        copied += _copy_prefix(src_sd, dst_sd, f"model.{src_idx}.", [f"model.{dst_idx}."], f"{tag} RGB neck {src_idx}->{dst_idx}")
    for src_idx, dst_idx in layout["ir_neck"].items():
        copied += _copy_prefix(src_sd, dst_sd, f"model.{src_idx}.", [f"model.{dst_idx}."], f"{tag} IR neck {src_idx}->{dst_idx}")

    copied += _copy_obb_head(src_sd, dst_sd, layout["head"], f"OBB 22->IAFAOBB {layout['head']}")
    return copied


def _initialize_iafa_before_fpn_target(src_sd: dict, dst_sd: dict, layout: dict, tag: str) -> int:
    """Initialize IAFA-before-FPN layouts with single-stream neck and OBB head."""
    copied = _copy_standard_backbone_to_last_dmaf(src_sd, dst_sd, layout)

    for src_idx, rgb_idx, ir_idx in layout.get("sppf_pairs", ()):
        copied += _copy_prefix(
            src_sd,
            dst_sd,
            f"model.{src_idx}.",
            [f"model.{rgb_idx}.", f"model.{ir_idx}."],
            f"SPPF {src_idx}->{rgb_idx},{ir_idx}",
        )

    for src_idx, dst_idx in layout["neck"].items():
        copied += _copy_prefix(src_sd, dst_sd, f"model.{src_idx}.", [f"model.{dst_idx}."], f"{tag} neck {src_idx}->{dst_idx}")

    copied += _copy_obb_head(src_sd, dst_sd, layout["head"], f"OBB 22->OBB {layout['head']}")
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Build two-stream OBB checkpoint from single-stream OBB checkpoint.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Single-stream .pt checkpoint path.")
    parser.add_argument("--target-yaml", type=Path, default=DEFAULT_TARGET_YAML, help="Two-stream model YAML path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output two-stream .pt path.")
    parser.add_argument(
        "--architecture",
        choices=(
            "auto",
            "legacy",
            "mbnet",
            "mbnet_expanded",
            "assa_dmaf_iafa",
            "assa_lastdmaf_fasterir_p2c2f",
            "assa_iafa_before_fpn_lastdmaf_fasterir_p2c2f",
        ),
        default="auto",
        help="Weight mapping layout. 'auto' detects supported two-stream YAML layouts by module names.",
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
    torch.load = _torch_load_trusted_checkpoint
    single = YOLO(str(args.source), task="obb")
    src_sd = single.model.state_dict()

    print(f"[INFO] building two-stream model from: {args.target_yaml}")
    two = YOLO(str(args.target_yaml), task="obb")
    dst_sd = two.model.state_dict()

    architecture = args.architecture
    if architecture == "auto":
        if "DualC2fLastDMAF_FasterIR" in yaml_text and "IAFAFusion" in yaml_text:
            architecture = "assa_iafa_before_fpn_lastdmaf_fasterir_p2c2f"
        elif "DualC2fLastDMAF_FasterIR" in yaml_text and "IAFAOBB" in yaml_text:
            architecture = "assa_lastdmaf_fasterir_p2c2f"
        elif "ASSARIFusion" in yaml_text and "DualC2fDMAF" in yaml_text and "IAFAOBB" in yaml_text:
            architecture = "assa_dmaf_iafa"
        elif "DualFeatureSelect" in yaml_text and "DualC2fDMAF" in yaml_text and "IAFAOBB" in yaml_text:
            architecture = "mbnet_expanded"
        elif "DualC2fDMAF" in yaml_text and "IAFAOBB" in yaml_text:
            architecture = "mbnet"
        else:
            architecture = "legacy"
    print(f"[INFO] mapping architecture: {architecture}")

    if architecture == "mbnet":
        copied = _initialize_mbnet_target(src_sd, dst_sd)
        print(f"[mbnet total] copied={copied}")
    elif architecture == "mbnet_expanded":
        copied = _initialize_expanded_iafa_target(src_sd, dst_sd, MBNET_EXPANDED_LAYOUT, "mbnet_expanded")
        print(f"[mbnet_expanded total] copied={copied}")
    elif architecture == "assa_dmaf_iafa":
        copied = _initialize_expanded_iafa_target(src_sd, dst_sd, ASSA_DMAF_IAFA_LAYOUT, "assa_dmaf_iafa")
        print(f"[assa_dmaf_iafa total] copied={copied}")
    elif architecture == "assa_lastdmaf_fasterir_p2c2f":
        copied = _initialize_last_dmaf_iafa_target(
            src_sd,
            dst_sd,
            ASSA_LASTDMAF_FASTIR_P2C2F_LAYOUT,
            "assa_lastdmaf_fasterir_p2c2f",
        )
        print(f"[assa_lastdmaf_fasterir_p2c2f total] copied={copied}")
    elif architecture == "assa_iafa_before_fpn_lastdmaf_fasterir_p2c2f":
        copied = _initialize_iafa_before_fpn_target(
            src_sd,
            dst_sd,
            ASSA_IAFA_BEFORE_FPN_LASTDMAF_FASTIR_P2C2F_LAYOUT,
            "assa_iafa_before_fpn_lastdmaf_fasterir_p2c2f",
        )
        print(f"[assa_iafa_before_fpn_lastdmaf_fasterir_p2c2f total] copied={copied}")
    else:
        # Legacy ASSA/RIFusion layouts retained for reproducibility of existing experiments.
        _copy_with_layer_map(src_sd, dst_sd, LEGACY_SINGLE_TO_TWOSTREAM_RGB, "single->twostream_rgb")
        rgb_sd = {k: v for k, v in dst_sd.items()}
        _copy_with_layer_map(rgb_sd, dst_sd, LEGACY_TWOSTREAM_RGB_TO_IR, "twostream_rgb->ir")

    missing_keys, unexpected_keys = two.model.load_state_dict(dst_sd, strict=False)
    print(f"[load_state_dict] missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")
    for module in two.model.modules():
        if hasattr(module, "clear_runtime_cache"):
            module.clear_runtime_cache()

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
