#!/usr/bin/env python3
"""Run a small DroneVehicle ablation: channel sparse attention vs spatial sparse+dense attention.

The script creates deterministic train/val/test file lists from data/dronevehicle.yaml, then
prints or runs two comparable YOLO OBB experiments:
  1) ASSARIFusion: original channel sparse attention branch.
  2) ASSADualBranchRIFusion: new spatial sparse+dense cross-modal branch.

Example:
  python tools/run_dronevehicle_attention_ablation.py --run --device 0 --epochs 30 --train-samples 2000 --val-samples 500
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ultralytics.utils import yaml_load, yaml_save


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "dronevehicle.yaml"
DEFAULT_CHANNEL_YAML = ROOT / "yaml" / "yolov8_twostream_obb_assafusion_postc2f.yaml"
DEFAULT_SPATIAL_YAML = ROOT / "yaml" / "yolov8_twostream_obb_assafusion_dualbranch_postc2f.yaml"
DEFAULT_WEIGHTS = ROOT / "pre-trained" / "yolov8s-obb_twostream.pt"


def resolve_split_path(data: dict, key: str) -> Path:
    """Resolve a split path in the same way as the local Ultralytics dataset checker."""
    value = data[key]
    path = Path(data.get("path", ""))
    split = Path(value)
    return split if split.is_absolute() else path / split


def list_images(path: Path) -> list[Path]:
    """Return sorted image files from a directory or an existing txt list."""
    suffixes = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
    if path.is_file():
        parent = path.parent
        files = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            files.append(p if p.is_absolute() else parent / p)
        return sorted(p for p in files if p.suffix.lower() in suffixes)
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes)
    raise FileNotFoundError(f"split path not found: {path}")


def sample_pairs(rgb_files: list[Path], ir_files: list[Path], n: int | None, seed: int) -> tuple[list[Path], list[Path]]:
    """Sample paired RGB/IR paths by sorted index, matching this repo's paired dataloader behavior."""
    if len(rgb_files) != len(ir_files):
        raise ValueError(f"RGB/IR count mismatch: {len(rgb_files)} vs {len(ir_files)}")
    indexes = list(range(len(rgb_files)))
    rng = random.Random(seed)
    rng.shuffle(indexes)
    if n and n > 0:
        indexes = indexes[: min(n, len(indexes))]
    indexes = sorted(indexes)
    return [rgb_files[i] for i in indexes], [ir_files[i] for i in indexes]


def write_list(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in files) + "\n", encoding="utf-8")


def build_subset_yaml(args: argparse.Namespace) -> Path:
    """Create sampled RGB/IR file lists and a matching data YAML."""
    data = yaml_load(args.data)
    out_dir = args.work_dir / f"seed{args.seed}_tr{args.train_samples}_val{args.val_samples}"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_specs = {
        "train": ("train", "train_ir", args.train_samples, args.seed),
        "val": ("val", "val_ir", args.val_samples, args.seed + 1),
        "test": ("test", "test_ir", args.test_samples, args.seed + 2),
    }
    subset = {
        "path": "",
        "names": data["names"],
        "nc": len(data["names"]),
    }

    for split_name, (rgb_key, ir_key, count, seed) in split_specs.items():
        if rgb_key not in data or ir_key not in data:
            continue
        rgb_files = list_images(resolve_split_path(data, rgb_key))
        ir_files = list_images(resolve_split_path(data, ir_key))
        rgb_sample, ir_sample = sample_pairs(rgb_files, ir_files, count, seed)
        rgb_list = out_dir / f"{split_name}_rgb.txt"
        ir_list = out_dir / f"{split_name}_ir.txt"
        write_list(rgb_list, rgb_sample)
        write_list(ir_list, ir_sample)
        subset[split_name] = str(rgb_list.resolve())
        subset[f"{split_name}_ir"] = str(ir_list.resolve())
        print(f"[subset] {split_name}: {len(rgb_sample)} pairs")

    yaml_path = out_dir / "dronevehicle_ablation.yaml"
    yaml_save(yaml_path, subset)
    print(f"[subset] yaml: {yaml_path}")
    return yaml_path


def train_and_test(exp_name: str, model_yaml: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    """Train one branch and immediately evaluate its best checkpoint on the test split."""
    from ultralytics import YOLO
    import ultralytics.nn.tasks  # noqa: F401

    model = YOLO(str(model_yaml))
    if args.weights and args.weights.exists():
        model.load(str(args.weights))
    elif args.weights:
        print(f"[WARN] weights not found, training from model init: {args.weights}")

    project = args.project / exp_name
    model.train(
        data=str(data_yaml),
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

    best = project / "train" / "weights" / "best.pt"
    if best.exists():
        YOLO(str(best)).val(
            data=str(data_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.val_batch,
            device=args.device,
            project=str(project / "train"),
            name="test_result",
            conf=args.conf,
            iou=args.iou,
            task="obb",
            exist_ok=True,
        )
    else:
        print(f"[WARN] best checkpoint not found: {best}")


def print_commands(data_yaml: Path, args: argparse.Namespace) -> None:
    """Print reproducible commands when the user wants to run manually."""
    common = (
        f"--data {data_yaml} --weights {args.weights} --epochs {args.epochs} --imgsz {args.imgsz} "
        f"--batch {args.batch} --device {args.device} --seed {args.seed}"
    )
    print("\nManual equivalent:")
    print(f"python tools/run_dronevehicle_attention_ablation.py --run --only channel {common}")
    print(f"python tools/run_dronevehicle_attention_ablation.py --run --only spatial {common}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DroneVehicle channel-vs-spatial attention ablation.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--channel-yaml", type=Path, default=DEFAULT_CHANNEL_YAML)
    parser.add_argument("--spatial-yaml", type=Path, default=DEFAULT_SPATIAL_YAML)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "ablation_data" / "dronevehicle_attention")
    parser.add_argument("--project", type=Path, default=ROOT / "runs_ablation" / "dronevehicle_attention")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--run", action="store_true", help="Actually train/evaluate. Without this, only create YAML.")
    parser.add_argument("--only", choices=("both", "channel", "spatial"), default="both")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_yaml = build_subset_yaml(args)
    if not args.run:
        print_commands(data_yaml, args)
        return 0

    if args.only in {"both", "channel"}:
        train_and_test("channel_sparse_assarifusion", args.channel_yaml, data_yaml, args)
    if args.only in {"both", "spatial"}:
        train_and_test("spatial_sparse_dense_dualbranch", args.spatial_yaml, data_yaml, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
