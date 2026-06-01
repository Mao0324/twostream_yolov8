#!/usr/bin/env python3
"""Run five DroneVehicle mAP50-oriented architecture ablations.

默认实验：
  baseline: 原 yolov8_twostream_obb_assafusion_postc2f.yaml
  A: 轻量 P2 head，P2 双流 ADD 后进入检测头
  B: P3/P4 前景门控 Dense 残差融合，P5 原 ASSA
  C: P3-only Dense-only 残差融合，P4/P5 原 ASSA
  D: P3/P4 Sparse+Dense 残差融合，P5 原 ASSA

训练/验证使用已有 dronevehicle_ablation.yaml，测试使用 data/dronevehicle.yaml 的完整 test split。
每个实验的完整测试日志会保存到 runs_ablation/dronevehicle_map50/<exp>/train/test_result/test.txt。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TRAIN_DATA = ROOT / "ablation_data" / "dronevehicle_attention" / "seed0_tr2000_val500" / "dronevehicle_ablation.yaml"
DEFAULT_TEST_DATA = ROOT / "data" / "dronevehicle.yaml"
DEFAULT_SOURCE_WEIGHTS = ROOT / "pre-trained" / "yolov8s-obb.pt"
DEFAULT_PROJECT = ROOT / "runs_ablation" / "dronevehicle_map50"
DEFAULT_WEIGHT_DIR = ROOT / "pre-trained" / "map50_ablation"


@dataclass(frozen=True)
class Experiment:
    name: str
    yaml_path: Path
    weight_path: Path


def build_experiments(weight_dir: Path) -> list[Experiment]:
    """五组结构消融配置，名称会同时作为 runs 目录名。"""
    return [
        Experiment(
            "baseline_assafusion",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_postc2f.yaml",
            weight_dir / "baseline_assafusion.pt",
        ),
        Experiment(
            "A_p2head_assafusion",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_p2head_postc2f.yaml",
            weight_dir / "A_p2head_assafusion.pt",
        ),
        Experiment(
            "B_fgated_dense_p3p4_p5assa",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_fgated_dense_p3p4_p5assa_postc2f.yaml",
            weight_dir / "B_fgated_dense_p3p4_p5assa.pt",
        ),
        Experiment(
            "C_denseonly_p3_p4p5assa",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_denseonly_p3_p4p5assa_postc2f.yaml",
            weight_dir / "C_denseonly_p3_p4p5assa.pt",
        ),
        Experiment(
            "D_residual_sd_p3p4_p5assa",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_residual_sd_p3p4_p5assa_postc2f.yaml",
            weight_dir / "D_residual_sd_p3p4_p5assa.pt",
        ),
    ]


def make_twostream_weights(source: Path, target_yaml: Path, output: Path, force: bool) -> None:
    """为每个结构生成一份匹配 YAML 的迁移权重，避免直接加载时大量 shape mismatch。"""
    if output.exists() and not force:
        print(f"[make-weights] skip existing: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "make_twostream_obb_weights.py"),
        "--source",
        str(source),
        "--target-yaml",
        str(target_yaml),
        "--output",
        str(output),
    ]
    print("[make-weights] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_one(exp: Experiment, args: argparse.Namespace) -> Path:
    """训练单个实验，返回 best.pt 路径。"""
    from ultralytics import YOLO
    import ultralytics.nn.tasks  # noqa: F401

    model = YOLO(str(exp.yaml_path))
    if exp.weight_path.exists():
        model.load(str(exp.weight_path))
    else:
        print(f"[WARN] transferred weights not found, train from yaml init: {exp.weight_path}")

    project = args.project / exp.name
    print(f"[train] {exp.name}")
    model.train(
        data=str(args.train_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name="train",
        task="obb",
        seed=args.seed,
        deterministic=True,
        close_mosaic=args.close_mosaic,
        cache=False,
        exist_ok=True,
    )
    # 释放训练模型引用，避免随后完整 test 子进程启动时 GPU 显存被当前进程继续占用。
    try:
        import gc
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - 清理失败不影响实验主流程
        print(f"[WARN] cuda cleanup skipped: {exc}")
    return project / "train" / "weights" / "best.pt"


def test_one(exp: Experiment, best: Path, args: argparse.Namespace) -> None:
    """用完整 DroneVehicle test split 测试，并把 stdout/stderr 保存到 test.txt。"""
    if not best.exists():
        print(f"[WARN] best checkpoint not found, skip test: {best}")
        return

    test_dir = args.project / exp.name / "train" / "test_result"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_txt = test_dir / "test.txt"

    cmd = [
        sys.executable,
        str(ROOT / "test_dronevehicle.py"),
        "--weights",
        str(best),
        "--data",
        str(args.test_data),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.val_batch),
        "--device",
        str(args.device),
        "--project",
        str(args.project / exp.name / "train"),
        "--name",
        "test_result",
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
    ]

    print(f"[test] {exp.name}")
    print("[test] " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("COMET_MODE", "disabled")
    with test_txt.open("w", encoding="utf-8", errors="replace") as f:
        f.write("[command] " + " ".join(cmd) + "\n\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
    print(f"[test] saved: {test_txt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five mAP50-oriented DroneVehicle architecture ablations.")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA, help="Ablation train/val YAML.")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA, help="Full DroneVehicle test YAML.")
    parser.add_argument("--source-weights", type=Path, default=DEFAULT_SOURCE_WEIGHTS)
    parser.add_argument("--weight-dir", type=Path, default=DEFAULT_WEIGHT_DIR)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--force-make-weights", action="store_true")
    parser.add_argument("--skip-make-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only print the experiment order and paths.")
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="Comma-separated experiment names, or all. Example: --only baseline_assafusion,A_p2head_assafusion",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = build_experiments(args.weight_dir)
    if args.only != "all":
        selected = {name.strip() for name in args.only.split(",") if name.strip()}
        experiments = [exp for exp in experiments if exp.name in selected]
        missing = selected - {exp.name for exp in experiments}
        if missing:
            raise ValueError(f"unknown experiment name(s): {sorted(missing)}")

    print("[data] train/val:", args.train_data)
    print("[data] full test:", args.test_data)
    print("[project]", args.project)
    for exp in experiments:
        print(f"[exp] {exp.name}: {exp.yaml_path}")

    if args.dry_run:
        return 0

    if not args.train_data.exists():
        raise FileNotFoundError(f"train data yaml not found: {args.train_data}")
    if not args.test_data.exists():
        raise FileNotFoundError(f"test data yaml not found: {args.test_data}")
    if not args.source_weights.exists() and not args.skip_make_weights:
        raise FileNotFoundError(f"source weights not found: {args.source_weights}")

    for exp in experiments:
        if not exp.yaml_path.exists():
            raise FileNotFoundError(f"model yaml not found: {exp.yaml_path}")
        if not args.skip_make_weights:
            make_twostream_weights(args.source_weights, exp.yaml_path, exp.weight_path, args.force_make_weights)
        best = train_one(exp, args)
        test_one(exp, best, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
