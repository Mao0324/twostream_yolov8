from __future__ import annotations

import csv
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import ASSAFusion  # noqa: E402
import ultralytics.nn.tasks  # noqa: E402, F401


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_assa_module_forward() -> None:
    print("[1] ASSAFusion forward checks")
    cases = [(2, 128, 80, 80), (2, 256, 40, 40), (2, 512, 20, 20)]
    for b, c, h, w in cases:
        module = ASSAFusion(c, reduction=16, min_ch=8, max_ch=32)
        x_rgb = torch.randn(b, c, h, w)
        x_ir = torch.randn(b, c, h, w)
        y = module([x_rgb, x_ir])
        print(
            f"  in={(b, c, h, w)} out={tuple(y.shape)} c_mid={module.c_mid} "
            f"params={count_params(module)}"
        )


def format_model_output_shapes(y):
    if isinstance(y, tuple):
        out = []
        for i, yi in enumerate(y):
            if isinstance(yi, (list, tuple)):
                out.append((i, [tuple(t.shape) for t in yi if hasattr(t, "shape")]))
            elif hasattr(yi, "shape"):
                out.append((i, tuple(yi.shape)))
            else:
                out.append((i, str(type(yi))))
        return out
    if isinstance(y, list):
        return [tuple(t.shape) for t in y if hasattr(t, "shape")]
    if hasattr(y, "shape"):
        return tuple(y.shape)
    return str(type(y))


def test_model_build_and_forward(model_yaml: Path) -> None:
    print("[2] Model build + forward checks")
    model = YOLO(str(model_yaml), task="obb")
    model.info(verbose=False)

    x = torch.randn(1, 6, 640, 640)
    with torch.no_grad():
        y = model.model(x)
    print(f"  {model_yaml.name} output shapes: {format_model_output_shapes(y)}")


def write_image(path: Path, size: int = 128, gray: bool = False) -> None:
    if gray:
        img = np.random.randint(0, 255, (size, size), dtype=np.uint8)
    else:
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def make_toy_dataset(root: Path) -> Path:
    print("[3] Creating toy OBB dataset")
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "image" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # class + 4 corners (normalized): cls x1 y1 x2 y2 x3 y3 x4 y4
    label_line = "0 0.20 0.20 0.65 0.20 0.65 0.60 0.20 0.60\n"

    for i in range(2):  # two train pairs -> one batch with batch=2
        stem = f"train_{i:03d}"
        write_image(root / "images" / "train" / f"{stem}.jpg", size=128, gray=False)
        write_image(root / "image" / "train" / f"{stem}.png", size=128, gray=True)
        (root / "labels" / "train" / f"{stem}.txt").write_text(label_line, encoding="utf-8")

    for i in range(1):
        stem = f"val_{i:03d}"
        write_image(root / "images" / "val" / f"{stem}.jpg", size=128, gray=False)
        write_image(root / "image" / "val" / f"{stem}.png", size=128, gray=True)
        (root / "labels" / "val" / f"{stem}.txt").write_text(label_line, encoding="utf-8")

    data_yaml = root / "toy_obb_twostream.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "train_ir: image/train",
                "val_ir: image/val",
                "names:",
                "  0: car",
                "  1: truck",
                "  2: bus",
                "  3: van",
                "  4: freight_car",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  dataset yaml: {data_yaml}")
    return data_yaml


def csv_has_nan(csv_path: Path) -> bool:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for v in row:
                if v.strip().lower() == "nan":
                    return True
    return False


def smoke_train(model_yaml: Path, data_yaml: Path, project_dir: Path) -> None:
    print("[4] 1-epoch OBB smoke train")
    model = YOLO(str(model_yaml), task="obb")
    train_exc = None
    try:
        model.train(
            data=str(data_yaml),
            epochs=1,
            imgsz=128,
            batch=2,
            device="cpu",
            workers=0,
            project=str(project_dir),
            name="assa_smoke",
            exist_ok=True,
            task="obb",
            val=False,
            cache=False,
            amp=False,
            save=False,
            plots=False,
        )
    except Exception as e:  # PyTorch>=2.6 default weights_only=True may fail in final_eval strip_optimizer.
        train_exc = e
        msg = str(e)
        if "Weights only load failed" in msg:
            print("  training loop finished, final_eval checkpoint strip hit torch.load(weights_only=True) limitation.")
        else:
            raise

    csv_path = Path(model.trainer.csv)
    has_nan = csv_has_nan(csv_path)
    print(f"  results.csv: {csv_path}")
    print(f"  has_nan: {has_nan}")
    if train_exc is not None:
        print(f"  post-train warning: {type(train_exc).__name__}: {str(train_exc).splitlines()[0]}")
    if has_nan:
        raise RuntimeError("NaN detected in smoke training results.csv")


def main() -> None:
    repo_root = REPO_ROOT
    model_yaml = repo_root / "yaml" / "yolov8_twostream_assa4.yaml"
    baseline_yaml = repo_root / "yaml" / "PC2f_MPF_yolov8s.yaml"

    test_assa_module_forward()
    test_model_build_and_forward(model_yaml)
    test_model_build_and_forward(baseline_yaml)

    toy_root = repo_root / "tmp" / "assa_toy_dataset"
    toy_yaml = make_toy_dataset(toy_root)
    smoke_train(model_yaml, toy_yaml, repo_root / "tmp" / "assa_smoke_runs")

    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
