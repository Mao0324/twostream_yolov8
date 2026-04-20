#!/usr/bin/env python3
"""
随机抽样 DroneVehicle 测试集图像，绘制 GT 与预测框（OBB）并保存。

示例:
python tools/visualize_dronevehicle_test_pred_gt.py \
  --weights /path/to/best.pt \
  --data data/dronevehicle.yaml \
  --num-samples 10 \
  --device 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Visualize DroneVehicle test predictions and GT.")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights, e.g. best.pt")
    parser.add_argument(
        "--data",
        type=str,
        default=str(repo_root / "data" / "dronevehicle.yaml"),
        help="Dataset yaml path",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Randomly selected test images")
    parser.add_argument("--imgsz", type=int, default=704, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id, e.g. 0 or 0,1")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(repo_root / "vis" / "dronevehicle_test_vis"),
        help="Output directory",
    )
    parser.add_argument("--line-width", type=int, default=2, help="Polyline thickness")
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def load_test_paths(data_yaml: Path) -> tuple[Path, Path, Path, dict[int, str]]:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"]).resolve()
    test_img_dir = resolve_path(root, cfg["test"])
    test_ir_dir = resolve_path(root, cfg["test_ir"])
    test_label_dir = test_img_dir.parent.parent / "labels" / test_img_dir.name

    names_raw = cfg.get("names", {})
    names = {int(k): str(v) for k, v in names_raw.items()}
    return test_img_dir, test_ir_dir, test_label_dir, names


def read_image_3ch(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def read_gt_polygons(label_path: Path, image_w: int, image_h: int) -> list[tuple[int, np.ndarray]]:
    gts: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return gts

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]

        if len(nums) >= 8:
            coords = nums[:8]
            pts = np.array(
                [[coords[i] * image_w, coords[i + 1] * image_h] for i in range(0, 8, 2)],
                dtype=np.float32,
            )
            gts.append((cls_id, pts))
        elif len(nums) == 4:
            x, y, w, h = nums
            cx, cy = x * image_w, y * image_h
            bw, bh = w * image_w, h * image_h
            pts = np.array(
                [
                    [cx - bw / 2, cy - bh / 2],
                    [cx + bw / 2, cy - bh / 2],
                    [cx + bw / 2, cy + bh / 2],
                    [cx - bw / 2, cy + bh / 2],
                ],
                dtype=np.float32,
            )
            gts.append((cls_id, pts))
    return gts


def draw_polygon(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    text: str,
    line_width: int,
) -> None:
    pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i], isClosed=True, color=color, thickness=line_width)
    x0, y0 = pts_i[0, 0].tolist()
    y0 = max(y0 - 4, 12)
    cv2.putText(img, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    # 延迟导入，避免仅查看 --help 时触发环境依赖报错
    from ultralytics import YOLO
    import ultralytics.nn.tasks  # noqa: F401

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    test_img_dir, test_ir_dir, test_label_dir, names = load_test_paths(data_yaml)
    if not test_img_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")
    if not test_ir_dir.exists():
        raise FileNotFoundError(f"Test IR directory not found: {test_ir_dir}")
    if not test_label_dir.exists():
        raise FileNotFoundError(f"Test label directory not found: {test_label_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    all_images = sorted([p for p in test_img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not all_images:
        raise RuntimeError(f"No test images found in: {test_img_dir}")

    k = min(args.num_samples, len(all_images))
    chosen_images = random.sample(all_images, k=k)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    chosen_txt = save_dir / "selected_images.txt"
    chosen_txt.write_text("\n".join(str(p) for p in chosen_images), encoding="utf-8")

    model = YOLO(args.weights)

    for img_path in chosen_images:
        rgb_img = read_image_3ch(img_path)
        if rgb_img is None:
            print(f"[WARN] skip unreadable RGB image: {img_path}")
            continue

        ir_path = test_ir_dir / img_path.name
        ir_img = read_image_3ch(ir_path)
        if ir_img is None:
            print(f"[WARN] skip missing/unreadable IR image: {ir_path}")
            continue
        if ir_img.shape[:2] != rgb_img.shape[:2]:
            print(f"[WARN] skip shape mismatch: rgb={rgb_img.shape[:2]}, ir={ir_img.shape[:2]} for {img_path.name}")
            continue

        vis_img = rgb_img.copy()
        h, w = vis_img.shape[:2]

        label_path = test_label_dir / f"{img_path.stem}.txt"
        gt_items = read_gt_polygons(label_path, w, h)
        for cls_id, pts in gt_items:
            cls_name = names.get(cls_id, str(cls_id))
            draw_polygon(vis_img, pts, (0, 255, 0), f"GT {cls_name}", args.line_width)

        two_stream_img = np.concatenate((rgb_img, ir_img), axis=2)

        result = model.predict(
            source=two_stream_img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            task="obb",
            verbose=False,
        )[0]

        if result.obb is not None and len(result.obb) > 0:
            pred_polys = result.obb.xyxyxyxy.cpu().numpy()
            pred_cls = result.obb.cls.int().cpu().tolist()
            pred_conf = result.obb.conf.cpu().tolist()
            for pts, cls_id, conf in zip(pred_polys, pred_cls, pred_conf):
                cls_name = names.get(int(cls_id), str(cls_id))
                draw_polygon(vis_img, pts, (0, 0, 255), f"Pred {cls_name} {conf:.2f}", args.line_width)

        out_file = save_dir / f"{img_path.stem}_pred_gt.jpg"
        cv2.imwrite(str(out_file), vis_img)
        print(f"[OK] saved: {out_file}")

    print(f"[DONE] visualization saved to: {save_dir}")
    print(f"[DONE] selected image list: {chosen_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
