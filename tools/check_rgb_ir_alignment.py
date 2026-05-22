#!/usr/bin/env python3
"""Check pixel-level RGB/IR alignment for this two-stream YOLO fork.

This diagnostic is intentionally data-side only: it does not change labels,
images, or training code. It estimates whether paired RGB/IR images are
spatially aligned before they are stacked into the 6-channel tensor used by the
two-stream dataloader.

Example:
  python tools/check_rgb_ir_alignment.py --data data/dronevehicle.yaml --split train --sample 200
  python tools/check_rgb_ir_alignment.py --rgb /path/to/visible/images --ir /path/to/infrared/images
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}


@dataclass
class PairResult:
    rgb: Path
    ir: Path
    width: int
    height: int
    ir_width: int
    ir_height: int
    dx_px: float
    dy_px: float
    shift_px: float
    response: float
    ncc_before: float
    ncc_after: float
    status: str
    reason: str


def resolve_dataset_path(data: dict, key: str) -> Path | list[Path] | None:
    """Resolve a YOLO dataset YAML path key, supporting dirs, text files, and lists."""
    value = data.get(key)
    if not value:
        return None

    yaml_file = Path(data.get("yaml_file", "")).resolve()
    root = Path(data.get("path") or yaml_file.parent)
    if not root.is_absolute():
        root = (yaml_file.parent / root).resolve()

    def _resolve_one(item: str) -> Path:
        p = Path(item)
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    if isinstance(value, (list, tuple)):
        return [_resolve_one(str(x)) for x in value]
    return _resolve_one(str(value))


def collect_images(path: Path | list[Path]) -> list[Path]:
    """Collect image files from a directory, image-list txt file, or a list of both."""
    paths = path if isinstance(path, list) else [path]
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(x for x in p.rglob("*") if x.is_file())
        elif p.is_file():
            parent = p.parent
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                files.append((parent / line[2:]).resolve() if line.startswith("./") else Path(line).resolve())
        else:
            raise FileNotFoundError(f"Image path does not exist: {p}")
    return sorted(x for x in files if x.suffix.lower().lstrip(".") in IMG_FORMATS)


def build_pairs(rgb_files: Sequence[Path], ir_files: Sequence[Path], pair_mode: str) -> list[tuple[Path, Path]]:
    """Build RGB/IR pairs in the same way users usually prepare two-stream datasets."""
    if pair_mode == "sorted":
        return list(zip(sorted(rgb_files), sorted(ir_files)))

    ir_by_stem: dict[str, Path] = {}
    duplicate_ir: set[str] = set()
    for p in ir_files:
        if p.stem in ir_by_stem:
            duplicate_ir.add(p.stem)
        ir_by_stem[p.stem] = p
    if duplicate_ir:
        dup = ", ".join(sorted(duplicate_ir)[:10])
        raise ValueError(f"Duplicate IR stems found; cannot pair safely by stem: {dup}")

    pairs = [(rgb, ir_by_stem[rgb.stem]) for rgb in rgb_files if rgb.stem in ir_by_stem]
    if not pairs:
        raise ValueError("No RGB/IR pairs found. Use --pair-mode sorted if filenames intentionally differ.")
    return pairs


def sample_pairs(pairs: Sequence[tuple[Path, Path]], sample: int) -> list[tuple[Path, Path]]:
    """Take a deterministic, evenly spaced sample so repeated checks are comparable."""
    if sample <= 0 or sample >= len(pairs):
        return list(pairs)
    indexes = np.linspace(0, len(pairs) - 1, sample, dtype=int)
    return [pairs[int(i)] for i in indexes]


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def resize_for_eval(rgb: np.ndarray, ir: np.ndarray, resize_long: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Resize both modalities to a common evaluation shape and return scale factors.

    如果原始 RGB/IR 尺寸不同，这里只为了估计内容偏移而拉到同一尺寸；结果会另外标记
    shape_mismatch，因为训练时 6 通道拼接本身仍然要求两路图像尺寸一致。
    """
    h, w = rgb.shape[:2]
    if resize_long > 0 and max(h, w) > resize_long:
        scale = resize_long / float(max(h, w))
        eval_w, eval_h = max(8, round(w * scale)), max(8, round(h * scale))
    else:
        eval_w, eval_h = w, h

    rgb_eval = cv2.resize(rgb, (eval_w, eval_h), interpolation=cv2.INTER_AREA)
    ir_eval = cv2.resize(ir, (eval_w, eval_h), interpolation=cv2.INTER_AREA)
    sx = w / float(eval_w)
    sy = h / float(eval_h)
    return rgb_eval, ir_eval, sx, sy


def edge_map(gray: np.ndarray) -> np.ndarray:
    """Create a modality-robust edge/gradient map for RGB-vs-IR matching.

    直接比较灰度值容易受光照、热响应差异影响；边缘和梯度结构更接近“几何内容”，
    因此更适合用来判断两路图像是否对齐。
    """
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag -= float(mag.mean())
    std = float(mag.std())
    if std > 1e-6:
        mag /= std
    return mag.astype(np.float32)


def normalized_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 0.0
    return float((a * b).sum() / denom)


def estimate_alignment(
    rgb_path: Path,
    ir_path: Path,
    resize_long: int,
    max_shift: float,
    min_response: float,
    min_ncc: float,
) -> PairResult:
    rgb = load_gray(rgb_path)
    ir = load_gray(ir_path)
    h, w = rgb.shape[:2]
    ih, iw = ir.shape[:2]
    shape_mismatch = (h, w) != (ih, iw)

    rgb_eval, ir_eval, sx, sy = resize_for_eval(rgb, ir, resize_long)
    rgb_edge = edge_map(rgb_eval)
    ir_edge = edge_map(ir_eval)

    window = cv2.createHanningWindow((rgb_edge.shape[1], rgb_edge.shape[0]), cv2.CV_32F)
    (dx_eval, dy_eval), response = cv2.phaseCorrelate(rgb_edge, ir_edge, window)
    dx = float(dx_eval * sx)
    dy = float(dy_eval * sy)
    shift = float(math.hypot(dx, dy))

    ncc_before = normalized_corr(rgb_edge, ir_edge)
    matrix = np.float32([[1, 0, dx_eval], [0, 1, dy_eval]])
    shifted_ir = cv2.warpAffine(
        ir_edge,
        matrix,
        (ir_edge.shape[1], ir_edge.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ncc_after = normalized_corr(rgb_edge, shifted_ir)

    reasons: list[str] = []
    if shape_mismatch:
        reasons.append("shape_mismatch")
    if response < min_response and max(ncc_before, ncc_after) < min_ncc:
        reasons.append("low_confidence")
    if shift > max_shift:
        reasons.append("large_shift")

    if shape_mismatch or shift > max_shift:
        status = "MISALIGNED"
    elif reasons:
        status = "UNCERTAIN"
    else:
        status = "ALIGNED"

    return PairResult(
        rgb=rgb_path,
        ir=ir_path,
        width=w,
        height=h,
        ir_width=iw,
        ir_height=ih,
        dx_px=dx,
        dy_px=dy,
        shift_px=shift,
        response=float(response),
        ncc_before=ncc_before,
        ncc_after=ncc_after,
        status=status,
        reason=",".join(reasons) if reasons else "ok",
    )


def percentile(values: Iterable[float], q: float) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def write_csv(path: Path, results: Sequence[PairResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PairResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in results:
            row = r.__dict__.copy()
            row["rgb"] = str(row["rgb"])
            row["ir"] = str(row["ir"])
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate RGB/IR spatial alignment for two-stream datasets.")
    parser.add_argument("--data", type=Path, help="Dataset YAML with train/val/test and train_ir/val_ir/test_ir keys.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--rgb", type=Path, help="RGB image directory or image-list txt.")
    parser.add_argument("--ir", type=Path, help="IR image directory or image-list txt.")
    parser.add_argument("--pair-mode", choices=("stem", "sorted"), default="stem")
    parser.add_argument("--sample", type=int, default=200, help="Number of pairs to check; <=0 checks all pairs.")
    parser.add_argument("--resize-long", type=int, default=640, help="Resize long side for faster matching; <=0 disables.")
    parser.add_argument("--max-shift", type=float, default=8.0, help="Shift threshold in original-image pixels.")
    parser.add_argument("--min-response", type=float, default=0.08, help="Minimum phase-correlation confidence.")
    parser.add_argument("--min-ncc", type=float, default=0.04, help="Minimum edge NCC used to avoid false low-confidence warnings.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum detailed pair results to print.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path for all checked pairs.")
    args = parser.parse_args()

    if args.data:
        data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
        data["yaml_file"] = str(args.data.resolve())
        rgb_path = resolve_dataset_path(data, args.split)
        ir_path = resolve_dataset_path(data, f"{args.split}_ir")
        if rgb_path is None or ir_path is None:
            raise SystemExit(f"Dataset YAML must contain {args.split!r} and {args.split + '_ir'!r}.")
    else:
        if not args.rgb or not args.ir:
            raise SystemExit("Use either --data, or both --rgb and --ir.")
        rgb_path, ir_path = args.rgb.resolve(), args.ir.resolve()

    rgb_files = collect_images(rgb_path)
    ir_files = collect_images(ir_path)
    pairs = build_pairs(rgb_files, ir_files, args.pair_mode)
    checked_pairs = sample_pairs(pairs, args.sample)

    print(f"RGB source: {rgb_path}")
    print(f"IR source:  {ir_path}")
    print(f"RGB images: {len(rgb_files)}")
    print(f"IR images:  {len(ir_files)}")
    print(f"Pairs:      {len(pairs)}")
    print(f"Checked:    {len(checked_pairs)}")

    results = [
        estimate_alignment(
            rgb,
            ir,
            resize_long=args.resize_long,
            max_shift=args.max_shift,
            min_response=args.min_response,
            min_ncc=args.min_ncc,
        )
        for rgb, ir in checked_pairs
    ]

    counts = {name: sum(r.status == name for r in results) for name in ("ALIGNED", "UNCERTAIN", "MISALIGNED")}
    shifts = [r.shift_px for r in results]
    print("\nSummary:")
    print(f"  aligned:    {counts['ALIGNED']}")
    print(f"  uncertain:  {counts['UNCERTAIN']}")
    print(f"  misaligned: {counts['MISALIGNED']}")
    print(f"  shift p50/p90/max px: {percentile(shifts, 50):.2f} / {percentile(shifts, 90):.2f} / {max(shifts, default=0.0):.2f}")

    risky = [r for r in results if r.status != "ALIGNED"]
    if risky:
        print("\nRisky pairs:")
        for r in risky[: args.limit]:
            print(
                f"  {r.status:10s} shift=({r.dx_px:.2f},{r.dy_px:.2f}) |mag|={r.shift_px:.2f}px "
                f"resp={r.response:.3f} ncc={r.ncc_before:.3f}->{r.ncc_after:.3f} reason={r.reason}"
            )
            print(f"    RGB: {r.rgb}")
            print(f"    IR:  {r.ir}")

    if args.csv:
        write_csv(args.csv, results)
        print(f"\nCSV written: {args.csv.resolve()}")

    # 返回码用于流水线检查：只要出现明显不对齐，就让命令失败；低置信度仅提示人工复核。
    return 1 if counts["MISALIGNED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
