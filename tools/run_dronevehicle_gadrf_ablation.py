#!/usr/bin/env python3
"""Run DroneVehicle GADRF architecture ablations.

Experiments:
  C_baseline_denseonly_p3_p4p5assa:
    P3 DenseResidual, P4/P5 sparse ASSA.
  P3_gadrf_p4p5assa:
    P3 Geometry-Aware Dense Residual RGB-IR Fusion, P4/P5 sparse ASSA.
  P3P4_gadrf_p5assa:
    P3 GADRF, P4 ARLocal + ASSA, P5 sparse ASSA.
  P3P4_gadrf_noaffine_p5assa:
    Same as P3P4_gadrf_p5assa, but ARLocal affine modulation is disabled.
  P3P4_gadrf_independent_p5assa:
    Same as P3P4_gadrf_p5assa, but RGB/IR use independent geometry gates.
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
DEFAULT_PROJECT = ROOT / "runs_ablation" / "dronevehicle_gadrf"
DEFAULT_WEIGHT_DIR = ROOT / "pre-trained" / "gadrf_ablation"


@dataclass(frozen=True)
class Experiment:
    name: str
    yaml_path: Path
    weight_path: Path


def build_experiments(weight_dir: Path) -> list[Experiment]:
    return [
        Experiment(
            "C_baseline_denseonly_p3_p4p5assa",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_denseonly_p3_p4p5assa_postc2f.yaml",
            weight_dir / "C_baseline_denseonly_p3_p4p5assa.pt",
        ),
        Experiment(
            "P3_gadrf_p4p5assa",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_gadrf_p3_p4p5assa_postc2f.yaml",
            weight_dir / "P3_gadrf_p4p5assa.pt",
        ),
        Experiment(
            "P3P4_gadrf_p5assa",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_gadrf_p3_gaassa_p4_p5assa_postc2f.yaml",
            weight_dir / "P3P4_gadrf_p5assa.pt",
        ),
        Experiment(
            "P3P4_gadrf_noaffine_p5assa",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_gadrf_p3_gaassa_p4_p5assa_noaffine_postc2f.yaml",
            weight_dir / "P3P4_gadrf_noaffine_p5assa.pt",
        ),
        Experiment(
            "P3P4_gadrf_independent_p5assa",
            ROOT / "yaml" / "yolov8_twostream_obb_assafusion_gadrf_p3_gaassa_p4_p5assa_independent_postc2f.yaml",
            weight_dir / "P3P4_gadrf_independent_p5assa.pt",
        ),
    ]


def make_twostream_weights(source: Path, target_yaml: Path, output: Path, force: bool) -> None:
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
    try:
        import gc
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"[WARN] cuda cleanup skipped: {exc}")
    return project / "train" / "weights" / "best.pt"


def test_one(exp: Experiment, best: Path, args: argparse.Namespace) -> None:
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
    parser = argparse.ArgumentParser(description="Run DroneVehicle GADRF ablations.")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--source-weights", type=Path, default=DEFAULT_SOURCE_WEIGHTS)
    parser.add_argument("--weight-dir", type=Path, default=DEFAULT_WEIGHT_DIR)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--force-make-weights", action="store_true")
    parser.add_argument("--skip-make-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", type=str, default="all", help="Comma-separated experiment names, or all.")
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
