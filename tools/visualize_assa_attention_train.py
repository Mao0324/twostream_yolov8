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


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


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
    parser.add_argument("--max-scan", type=int, default=5000, help="Maximum RGB files to scan before sampling")
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


def find_ir_pair(ir_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTS:
        candidate = ir_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def collect_image_pairs(rgb_dir: Path, ir_dir: Path, num_samples: int, seed: int, max_scan: int) -> list[tuple[Path, Path]]:
    rng = random.Random(seed)
    pairs: list[tuple[Path, Path]] = []
    seen_pairs = 0
    scanned = 0

    for rgb_path in rgb_dir.iterdir():
        if max_scan > 0 and scanned >= max_scan:
            break
        if not rgb_path.is_file() or rgb_path.suffix.lower() not in IMAGE_EXTS:
            continue

        scanned += 1
        ir_path = find_ir_pair(ir_dir, rgb_path.stem)
        if ir_path is None:
            continue

        # 使用 reservoir sampling，只扫描前 max_scan 个 RGB 文件，不需要遍历完整训练集。
        seen_pairs += 1
        if len(pairs) < num_samples:
            pairs.append((rgb_path, ir_path))
        else:
            replace_idx = rng.randrange(seen_pairs)
            if replace_idx < num_samples:
                pairs[replace_idx] = (rgb_path, ir_path)

    print(f"[INFO] scanned_rgb={scanned}, paired={seen_pairs}, selected={len(pairs)}")
    return pairs


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    # 延迟导入 torch/ultralytics，方便在没有深度学习环境时仍可查看脚本参数。
    from ultralytics import YOLO
    import ultralytics.nn.tasks  # noqa: F401
    import torch
    import torch.nn.functional as F
    from ultralytics.nn.modules.block import SparseCrossChannelAttention2d

    data_yaml = Path(args.data).resolve()
    rgb_dir, ir_dir = load_split_dirs(data_yaml, args.split)
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not ir_dir.exists():
        raise FileNotFoundError(f"IR directory not found: {ir_dir}")

    candidates = collect_image_pairs(rgb_dir, ir_dir, args.num_samples, args.seed, args.max_scan)
    if not candidates:
        raise RuntimeError(f"No paired images found by filename stem in first {args.max_scan} RGB files of {rgb_dir}")
    chosen = candidates

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "selected_pairs.txt").write_text(
        "\n".join(f"{rgb_path}\t{ir_path}" for rgb_path, ir_path in chosen),
        encoding="utf-8",
    )

    model = YOLO(args.weights)
    records: list[tuple[str, np.ndarray, float]] = []

    def patch_attention_capture(module: SparseCrossChannelAttention2d) -> None:
        if hasattr(module, "last_attn"):
            return

        module.last_attn = None
        module.last_sparsity = None

        def forward_with_capture(self, src, ref):
            b, _, h, w = src.shape
            q = self.q_proj(self.norm_src(src))
            q = q + self.q_dw(q)
            kv = self.kv_proj(self.norm_ref(ref))
            kv = kv + self.kv_dw(kv)

            q = q.reshape(b, self.heads, self.dim, h * w)
            v = kv.reshape(b, self.heads, self.dim, h * w)
            k = F.normalize(v, dim=-1)
            q = F.normalize(q, dim=-1)

            attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
            attn = F.relu(attn)
            if self.norm_attn:
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
            self.last_sparsity = (attn <= 1e-6).float().mean().detach()
            self.last_attn = attn.detach()

            out = torch.matmul(attn, v).reshape(b, self.hidden, h, w)
            return self.out_proj(out)

        # 兼容旧代码训练出的模型：旧模块没有 last_attn，这里只在可视化时替换 forward 以捕获注意力矩阵。
        module.forward = forward_with_capture.__get__(module, module.__class__)

    def make_hook(name: str):
        def hook(module, _inputs, _output):
            if not hasattr(module, "last_attn") or module.last_attn is None:
                return
            attn = module.last_attn[0].detach().float().cpu().numpy()
            sparsity = float(module.last_sparsity.detach().float().cpu().item()) if module.last_sparsity is not None else 0.0
            records.append((name, attn, sparsity))

        return hook

    handles = []
    for name, module in model.model.named_modules():
        if isinstance(module, SparseCrossChannelAttention2d):
            patch_attention_capture(module)
            handles.append(module.register_forward_hook(make_hook(name)))

    expected_records = len(handles) * 2
    if expected_records == 0:
        raise RuntimeError("No SparseCrossChannelAttention2d modules found in model.")

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
            image_records = records[-expected_records:]

            header = hstack_same_height([resize_with_title(rgb, "RGB"), resize_with_title(ir, "IR")])
            heatmaps = []
            per_layer_count: dict[str, int] = {}
            for module_name, attn, relu_sparsity in image_records:
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
