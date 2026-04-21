#!/usr/bin/env python3
"""Migrate old two-stream checkpoint weights to a new two-stream ASSA architecture.

Default paths follow the user's requested locations:
  old yaml:
    /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs/train2/PC2f_MPF_yolov8s.yaml
  old best checkpoint:
    /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs/train2/weights/best.pt
  new yaml:
    /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs/train3/PC2f_MPF_yolov8s.yaml
  output checkpoint:
    /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream_assa_pre.pt

Usage:
  python tools/migrate_twostream_weights_to_assa.py
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


DEFAULT_OLD_YAML = Path(
    "/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs/train2/PC2f_MPF_yolov8s.yaml"
)
DEFAULT_OLD_WEIGHTS = Path(
    "/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs/train2/weights/best.pt"
)
DEFAULT_NEW_YAML = Path(
    "/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs/train3/PC2f_MPF_yolov8s.yaml"
)
DEFAULT_OUTPUT = Path(
    "/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream_assa_pre.pt"
)


def transfer_same_name_same_shape(src_sd: dict[str, torch.Tensor], dst_sd: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Copy parameters from src to dst when key and shape are both matched."""
    copied = 0
    not_found = 0
    shape_mismatch = 0

    for k, v_dst in dst_sd.items():
        v_src = src_sd.get(k)
        if v_src is None:
            not_found += 1
            continue
        if v_src.shape != v_dst.shape:
            shape_mismatch += 1
            continue
        dst_sd[k] = v_src.clone()
        copied += 1

    return copied, not_found, shape_mismatch


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate old checkpoint weights to new twostream ASSA architecture.")
    parser.add_argument("--old-yaml", type=Path, default=DEFAULT_OLD_YAML, help="Old model yaml path.")
    parser.add_argument("--old-weights", type=Path, default=DEFAULT_OLD_WEIGHTS, help="Old checkpoint path (best.pt).")
    parser.add_argument("--new-yaml", type=Path, default=DEFAULT_NEW_YAML, help="New model yaml path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output migrated checkpoint path.")
    parser.add_argument("--task", type=str, default="obb", help="YOLO task, default: obb")
    args = parser.parse_args()

    for p, name in [
        (args.old_yaml, "old-yaml"),
        (args.old_weights, "old-weights"),
        (args.new_yaml, "new-yaml"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    print(f"[INFO] loading old checkpoint: {args.old_weights}")
    old_model = YOLO(str(args.old_weights), task=args.task)
    old_sd = old_model.model.state_dict()

    print(f"[INFO] building new model from yaml: {args.new_yaml}")
    new_model = YOLO(str(args.new_yaml), task=args.task)
    new_sd = new_model.model.state_dict()

    copied, not_found, shape_mismatch = transfer_same_name_same_shape(old_sd, new_sd)
    print(
        f"[MIGRATE] copied={copied}, dst_key_not_in_src={not_found}, shape_mismatch={shape_mismatch}, "
        f"dst_total={len(new_sd)}"
    )

    missing_keys, unexpected_keys = new_model.model.load_state_dict(new_sd, strict=False)
    print(f"[LOAD] missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "model": deepcopy(new_model.model).half(),
    }
    torch.save(ckpt, str(args.output))
    print(f"[DONE] saved migrated checkpoint to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
