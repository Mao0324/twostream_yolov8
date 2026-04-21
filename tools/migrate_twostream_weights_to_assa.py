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
from typing import Iterable

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


def _numel(sd: dict[str, torch.Tensor], keys: Iterable[str]) -> int:
    return int(sum(sd[k].numel() for k in keys))


def transfer_same_name_same_shape(
    src_sd: dict[str, torch.Tensor], dst_sd: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Copy src->dst weights by exact key and shape match, and collect report details."""
    dst_after = {k: v.clone() for k, v in dst_sd.items()}
    copied_keys: list[str] = []
    not_found_keys: list[str] = []
    shape_mismatch_keys: list[str] = []

    for k, v_dst in dst_sd.items():
        v_src = src_sd.get(k)
        if v_src is None:
            not_found_keys.append(k)
            continue
        if v_src.shape != v_dst.shape:
            shape_mismatch_keys.append(f"{k} | src={tuple(v_src.shape)} dst={tuple(v_dst.shape)}")
            continue
        dst_after[k] = v_src.clone()
        copied_keys.append(k)

    report = {
        "copied_keys": copied_keys,
        "not_found_keys": not_found_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }
    return dst_after, report


def _format_ratio(hit: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * hit / total):.2f}%"


def _print_list(title: str, items: list[str], limit: int) -> None:
    print(f"\n[{title}] total={len(items)}")
    if not items:
        return
    n = min(limit, len(items))
    for x in items[:n]:
        print(f"  - {x}")
    if len(items) > n:
        print(f"  ... ({len(items) - n} more)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate old checkpoint weights to new twostream ASSA architecture.")
    parser.add_argument("--old-yaml", type=Path, default=DEFAULT_OLD_YAML, help="Old model yaml path.")
    parser.add_argument("--old-weights", type=Path, default=DEFAULT_OLD_WEIGHTS, help="Old checkpoint path (best.pt).")
    parser.add_argument("--new-yaml", type=Path, default=DEFAULT_NEW_YAML, help="New model yaml path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output migrated checkpoint path.")
    parser.add_argument("--task", type=str, default="obb", help="YOLO task, default: obb")
    parser.add_argument("--list-limit", type=int, default=80, help="Max lines printed for each unmatched list.")
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Optional report txt path. Default: <output>.transfer_report.txt",
    )
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

    migrated_sd, report = transfer_same_name_same_shape(old_sd, new_sd)
    copied_keys = report["copied_keys"]
    not_found_keys = report["not_found_keys"]
    shape_mismatch_keys = report["shape_mismatch_keys"]

    copied_key_num = len(copied_keys)
    dst_total_keys = len(new_sd)
    copied_param_num = _numel(new_sd, copied_keys)
    dst_total_params = _numel(new_sd, new_sd.keys())

    print("\n========== Transfer Summary ==========")
    print(
        f"[KEYS ] copied={copied_key_num}/{dst_total_keys} "
        f"({_format_ratio(copied_key_num, dst_total_keys)})"
    )
    print(
        f"[PARAM] copied={copied_param_num}/{dst_total_params} "
        f"({_format_ratio(copied_param_num, dst_total_params)})"
    )
    print(
        f"[MISS ] dst_key_not_in_src={len(not_found_keys)}, "
        f"shape_mismatch={len(shape_mismatch_keys)}"
    )

    missing_keys, unexpected_keys = new_model.model.load_state_dict(migrated_sd, strict=False)
    print(f"[LOAD ] missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")

    _print_list("Not Found In Source (dst key not in src)", not_found_keys, args.list_limit)
    _print_list("Shape Mismatch", shape_mismatch_keys, args.list_limit)
    _print_list("load_state_dict Missing Keys", list(missing_keys), args.list_limit)
    _print_list("load_state_dict Unexpected Keys", list(unexpected_keys), args.list_limit)

    report_lines = [
        "========== Transfer Summary ==========",
        f"old_weights: {args.old_weights}",
        f"new_yaml: {args.new_yaml}",
        f"output: {args.output}",
        "",
        f"[KEYS ] copied={copied_key_num}/{dst_total_keys} ({_format_ratio(copied_key_num, dst_total_keys)})",
        f"[PARAM] copied={copied_param_num}/{dst_total_params} ({_format_ratio(copied_param_num, dst_total_params)})",
        f"[MISS ] dst_key_not_in_src={len(not_found_keys)}, shape_mismatch={len(shape_mismatch_keys)}",
        f"[LOAD ] missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}",
        "",
        "[Not Found In Source (dst key not in src)]",
        *not_found_keys,
        "",
        "[Shape Mismatch]",
        *shape_mismatch_keys,
        "",
        "[load_state_dict Missing Keys]",
        *list(missing_keys),
        "",
        "[load_state_dict Unexpected Keys]",
        *list(unexpected_keys),
        "",
    ]
    report_path = args.report_file or args.output.with_suffix(args.output.suffix + ".transfer_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[REPORT] saved: {report_path}")

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
