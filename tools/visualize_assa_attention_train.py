#!/usr/bin/env python3
"""可视化 ASSA 跨模态注意力矩阵在训练集样本上的稀疏聚合效果。

示例:
python tools/visualize_assa_attention_train.py \
  --weights /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/runs/train/weights/best.pt \
  --data data/dronevehicle.yaml \
  --num-samples 20 \
  --device 0
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Visualize ASSA attention matrices on training RGB/IR pairs.")
    parser.add_argument("--weights", type=str, required=True, help="Trained model weights, e.g. best.pt")
    parser.add_argument("--data", type=str, default=str(repo_root / "data" / "dronevehicle.yaml"), help="Dataset yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of RGB/IR pairs to sample")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device, e.g. 0 or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default=str(repo_root / "vis" / "assa_attention_train"), help="Output dir")
    parser.add_argument("--conf", type=float, default=0.001, help="Low conf keeps inference path active; boxes are not used")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_split_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"]).resolve()
    rgb_dir = resolve_path(root, cfg[split])
    ir_key = f"{split}_ir"
    if ir_key not in cfg:
        raise KeyError(f"Dataset yaml missing key: {ir_key}")
    ir_dir = resolve_path(root, cfg[ir_key])
    return rgb_dir, ir_dir


def read_image_3ch(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def resize_with_title(img: np.ndarray, title: str, width: int = 320) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = max(1, int(h * width / w))
    out = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def attention_to_heatmap(attn: np.ndarray, title: str, size: int = 320) -> np.ndarray:
    # attn 形状为 [heads, dim, dim]，这里对多头取均值，得到通道-通道注意力矩阵。
    matrix = attn.mean(axis=0)
    matrix = matrix - matrix.min()
    denom = matrix.max() + 1e-12
    matrix = (matrix / denom * 255).astype(np.uint8)
    matrix = cv2.resize(matrix, (size, size), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(matrix, cv2.COLORMAP_TURBO)
    cv2.putText(heatmap, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(heatmap, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 1, cv2.LINE_AA)
    return heatmap


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    pad = np.full((height - img.shape[0], img.shape[1], 3), 255, dtype=img.dtype)
    return np.vstack([img, pad])


def hstack_same_height(images: list[np.ndarray]) -> np.ndarray:
    height = max(img.shape[0] for img in images)
    return np.hstack([pad_to_height(img, height) for img in images])


def attention_stats(attn: np.ndarray) -> tuple[float, float, float]:
    prob = np.clip(attn.mean(axis=0), 0.0, None)
    sparsity = float((prob <= 1e-6).mean())
    row_sum = prob.sum(axis=-1, keepdims=True) + 1e-12
    prob = prob / row_sum
    top1 = float(prob.max(axis=-1).mean())
    entropy = float((-(prob * np.log(prob + 1e-12)).sum(axis=-1) / np.log(prob.shape[-1])).mean())
    return sparsity, top1, entropy


def collect_image_pairs(rgb_dir: Path, ir_dir: Path) -> list[tuple[Path, Path]]:
    rgb_paths = sorted(p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    ir_by_stem = {
        p.stem: p
        for p in sorted(ir_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }

    # DroneVehicle 双流目录中 RGB 常为 .jpg，IR 常为 .png，因此按 stem 配对而不是按完整文件名配对。
    return [(rgb_path, ir_by_stem[rgb_path.stem]) for rgb_path in rgb_paths if rgb_path.stem in ir_by_stem]


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    # 延迟导入 torch/ultralytics，方便在没有深度学习环境时仍可查看脚本参数。
    from ultralytics import YOLO
    import ultralytics.nn.tasks  # noqa: F401
    from ultralytics.nn.modules.block import SparseCrossChannelAttention2d

    data_yaml = Path(args.data).resolve()
    rgb_dir, ir_dir = load_split_dirs(data_yaml, args.split)
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not ir_dir.exists():
        raise FileNotFoundError(f"IR directory not found: {ir_dir}")

    candidates = collect_image_pairs(rgb_dir, ir_dir)
    if not candidates:
        raise RuntimeError(f"No paired images found by filename stem in {rgb_dir} and {ir_dir}")
    chosen = random.sample(candidates, k=min(args.num_samples, len(candidates)))

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "selected_pairs.txt").write_text(
        "\n".join(f"{rgb_path}\t{ir_path}" for rgb_path, ir_path in chosen),
        encoding="utf-8",
    )

    model = YOLO(args.weights)
    records: list[tuple[str, np.ndarray, float]] = []

    def make_hook(name: str):
        def hook(module, _inputs, _output):
            if module.last_attn is None:
                return
            attn = module.last_attn[0].detach().float().cpu().numpy()
            sparsity = float(module.last_sparsity.detach().float().cpu().item())
            records.append((name, attn, sparsity))

        return hook

    handles = []
    for name, module in model.model.named_modules():
        if isinstance(module, SparseCrossChannelAttention2d):
            handles.append(module.register_forward_hook(make_hook(name)))

    summary_rows = []
    directions = ("IR->RGB", "RGB->IR")
    try:
        for rgb_path, ir_path in chosen:
            rgb = read_image_3ch(rgb_path)
            ir = read_image_3ch(ir_path)
            if rgb.shape[:2] != ir.shape[:2]:
                print(f"[WARN] skip shape mismatch: {rgb_path.name}")
                continue

            records.clear()
            two_stream = np.concatenate([rgb, ir], axis=2)
            model.predict(source=two_stream, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)

            header = hstack_same_height([resize_with_title(rgb, "RGB"), resize_with_title(ir, "IR")])
            heatmaps = []
            per_layer_count: dict[str, int] = {}
            for module_name, attn, relu_sparsity in records:
                layer_name = module_name.split(".")[1] if module_name.startswith("model.") else module_name
                call_idx = per_layer_count.get(layer_name, 0)
                per_layer_count[layer_name] = call_idx + 1
                direction = directions[call_idx % 2]
                sparsity, top1, entropy = attention_stats(attn)
                title = f"L{layer_name} {direction} zero={relu_sparsity:.2f}"
                heatmaps.append(attention_to_heatmap(attn, title))
                summary_rows.append(
                    {
                        "image": rgb_path.name,
                        "layer": layer_name,
                        "direction": direction,
                        "relu_zero_ratio": f"{relu_sparsity:.6f}",
                        "matrix_zero_ratio": f"{sparsity:.6f}",
                        "row_top1_mean": f"{top1:.6f}",
                        "row_entropy_norm": f"{entropy:.6f}",
                    }
                )

            rows = [header]
            for i in range(0, len(heatmaps), 2):
                rows.append(hstack_same_height(heatmaps[i : i + 2]))
            canvas_width = max(row.shape[1] for row in rows)
            padded_rows = []
            for row in rows:
                if row.shape[1] < canvas_width:
                    pad = np.full((row.shape[0], canvas_width - row.shape[1], 3), 255, dtype=row.dtype)
                    row = np.hstack([row, pad])
                padded_rows.append(row)
            canvas = np.vstack(padded_rows)
            out_file = save_dir / f"{rgb_path.stem}_attention.jpg"
            cv2.imwrite(str(out_file), canvas)
            print(f"[OK] saved: {out_file}")
    finally:
        for handle in handles:
            handle.remove()

    csv_path = save_dir / "attention_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "layer", "direction", "relu_zero_ratio", "matrix_zero_ratio", "row_top1_mean", "row_entropy_norm"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[DONE] saved visualizations to: {save_dir}")
    print(f"[DONE] saved summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
