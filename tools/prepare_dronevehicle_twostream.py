#!/usr/bin/env python3
"""
准备 DroneVehicle 双流训练数据目录（等价于之前 train_dronevehicle.py 的 --prepare-only --force-rebuild 功能）。

目标：
把原始数据整理成该仓库双流加载器可直接使用的结构：

    <out_root>/
      images/{train,val,test}/ir_xxxxx.jpg   # 主流（RGB），但文件名使用 ir_ 前缀以匹配标签
      image/{train,val,test}/ir_xxxxx.(png|jpg|jpeg)  # 红外流
      labels/{train,val,test}/ir_xxxxx.txt   # OBB 标签

说明：
1) 原始标签通常是 ir_00001.txt，RGB 通常是 00001.jpg，IR 通常是 ir_00001.png。
2) 由于此仓库会按主流图像文件名自动寻找同名标签，所以这里把 RGB 输出命名为 ir_00001.jpg。
3) 默认使用软链接(symlink)，几乎不占空间；也支持硬链接或复制。



使用：python /home/ubuntu/MCONG/twostream_yolov8/tools/prepare_dronevehicle_twostream.py --force-rebuild

"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class SplitConfig:
    """一个数据划分（train/test/val）的输入配置。"""

    name: str
    image_dir: Path
    label_dir: Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Prepare DroneVehicle twostream dataset layout")

    # 原始图像目录（每个目录里同时包含 RGB 与 IR）
    parser.add_argument(
        "--train-images",
        default="/home/biiteam/Storage-4T/biiteam/WBY/datasets/DroneVehicle_SMFDet2/train_images/png_images",
        help="训练集图像目录（包含 RGB 与 IR）",
    )
    parser.add_argument(
        "--test-images",
        default="/home/biiteam/Storage-4T/biiteam/WBY/datasets/DroneVehicle_SMFDet2/test_images/png_images",
        help="测试集图像目录（包含 RGB 与 IR）",
    )
    parser.add_argument(
        "--val-images",
        default="/home/biiteam/Storage-4T/biiteam/WBY/datasets/DroneVehicle_SMFDet2/val_images/png_images",
        help="验证集图像目录（包含 RGB 与 IR）",
    )

    # 目标标签目录（已转好的 YOLO-OBB）
    parser.add_argument(
        "--train-labels",
        default="/home/biiteam/Storage-4T/biiteam/MCONG/datasets/dronevehiclelabels_obb/train_labels",
        help="训练集标签目录（ir_*.txt）",
    )
    parser.add_argument(
        "--test-labels",
        default="/home/biiteam/Storage-4T/biiteam/MCONG/datasets/dronevehiclelabels_obb/test_labels",
        help="测试集标签目录（ir_*.txt）",
    )
    parser.add_argument(
        "--val-labels",
        default="/home/biiteam/Storage-4T/biiteam/MCONG/datasets/dronevehiclelabels_obb/val_labels",
        help="验证集标签目录（ir_*.txt）",
    )

    # 输出根目录
    parser.add_argument(
        "--out-root",
        default="/home/biiteam/Storage-4T/biiteam/MCONG/datasets/dronevehicle_twostream",
        help="输出数据根目录",
    )

    # 文件落盘方式
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="输出文件创建方式：软链接/硬链接/复制",
    )

    # 是否删除并重建输出目录中的 split 子目录
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="删除并重建输出目录中的 train/test/val 子目录（等价之前 --force-rebuild）",
    )

    return parser.parse_args()


def _safe_mkdir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def _materialize(src: Path, dst: Path, mode: str) -> None:
    """
    按指定模式把 src 落到 dst。

    - symlink: 软链接（推荐，省空间）
    - hardlink: 硬链接
    - copy: 实际复制文件
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def _iter_ir_label_files(label_dir: Path) -> Iterable[Path]:
    """返回标签目录中的 ir_*.txt（稳定排序）。"""
    return sorted(label_dir.glob("ir_*.txt"))


def _pick_existing_file(candidates: List[Path]) -> Path | None:
    """从候选路径中选出第一个存在的文件。"""
    for p in candidates:
        if p.is_file():
            return p
    return None


def _find_rgb_file(image_dir: Path, rgb_stem: str) -> Path | None:
    """
    查找 RGB 文件。

    优先顺序：jpg > png > jpeg（同时兼容大写扩展名）。
    """
    candidates = [
        image_dir / f"{rgb_stem}.jpg",
        image_dir / f"{rgb_stem}.png",
        image_dir / f"{rgb_stem}.jpeg",
        image_dir / f"{rgb_stem}.JPG",
        image_dir / f"{rgb_stem}.PNG",
        image_dir / f"{rgb_stem}.JPEG",
    ]
    return _pick_existing_file(candidates)


def _find_ir_file(image_dir: Path, ir_stem: str) -> Path | None:
    """
    查找 IR 文件。

    兼容 ir_*.png / ir_*.jpg / ir_*.jpeg 及其大写扩展名。
    """
    candidates = [
        image_dir / f"{ir_stem}.png",
        image_dir / f"{ir_stem}.jpg",
        image_dir / f"{ir_stem}.jpeg",
        image_dir / f"{ir_stem}.PNG",
        image_dir / f"{ir_stem}.JPG",
        image_dir / f"{ir_stem}.JPEG",
    ]
    return _pick_existing_file(candidates)


def prepare_split(split: SplitConfig, out_root: Path, link_mode: str, force_rebuild: bool) -> Dict[str, int]:
    """
    准备一个 split（train/test/val）。

    返回统计：
      {
        "labels_total": int,   # 标签总数
        "prepared": int,       # 成功配对并输出
        "skipped": int,        # 失败或跳过
      }
    """
    if not split.image_dir.is_dir():
        raise FileNotFoundError(f"[{split.name}] image dir not found: {split.image_dir}")
    if not split.label_dir.is_dir():
        raise FileNotFoundError(f"[{split.name}] label dir not found: {split.label_dir}")

    out_rgb_dir = out_root / "images" / split.name
    out_ir_dir = out_root / "image" / split.name
    out_lb_dir = out_root / "labels" / split.name

    if force_rebuild:
        for d in (out_rgb_dir, out_ir_dir, out_lb_dir):
            if d.exists():
                shutil.rmtree(d)

    _safe_mkdir(out_rgb_dir)
    _safe_mkdir(out_ir_dir)
    _safe_mkdir(out_lb_dir)

    label_files = list(_iter_ir_label_files(split.label_dir))
    prepared = 0
    skipped = 0

    for lb in label_files:
        # 标签名要求 ir_*.txt
        ir_stem = lb.stem  # ir_00001
        if not ir_stem.startswith("ir_"):
            skipped += 1
            continue

        # 对应 RGB 文件名（去掉 ir_ 前缀）
        rgb_stem = ir_stem[3:]  # 00001
        rgb_src = _find_rgb_file(split.image_dir, rgb_stem)
        #ir_src = split.image_dir / f"{ir_stem}.png"
        #不写死 ir_*.png，改成自动找 png/jpg/jpeg，输出文件名后缀跟源 IR 一致（避免把 jpg 链接成 .png）
        ir_candidates = [
            split.image_dir / f"{ir_stem}.png",
            split.image_dir / f"{ir_stem}.jpg",
            split.image_dir / f"{ir_stem}.jpeg",
            ]
        ir_src = next((p for p in ir_candidates if p.is_file()), ir_candidates[0])

        if rgb_src is None or ir_src is None:
            skipped += 1
            print(
                f"[WARN] [{split.name}] missing pair for {lb.name}: "
                f"rgb_found={rgb_src is not None}, ir_found={ir_src is not None}"
            )
            continue

        # 主流输出固定为 .jpg，便于与 labels/ir_*.txt 一一对应（同 stem）。
        rgb_dst = out_rgb_dir / f"{ir_stem}.jpg"
        # IR 输出扩展名与源一致，避免 test 是 jpg 时出错。
        ir_dst = out_ir_dir / f"{ir_stem}{ir_src.suffix.lower()}"
        lb_dst = out_lb_dir / lb.name

        _materialize(rgb_src, rgb_dst, link_mode)
        _materialize(ir_src, ir_dst, link_mode)
        _materialize(lb, lb_dst, link_mode)
        prepared += 1

    return {"labels_total": len(label_files), "prepared": prepared, "skipped": skipped}


def main() -> int:
    """程序入口。"""
    args = parse_args()

    out_root = Path(args.out_root)
    _safe_mkdir(out_root)

    splits = [
        SplitConfig("train", Path(args.train_images), Path(args.train_labels)),
        SplitConfig("test", Path(args.test_images), Path(args.test_labels)),
        SplitConfig("val", Path(args.val_images), Path(args.val_labels)),
    ]

    total_labels = 0
    total_prepared = 0
    total_skipped = 0

    try:
        for split in splits:
            stats = prepare_split(
                split=split,
                out_root=out_root,
                link_mode=args.link_mode,
                force_rebuild=args.force_rebuild,
            )
            total_labels += stats["labels_total"]
            total_prepared += stats["prepared"]
            total_skipped += stats["skipped"]

            print(
                f"[INFO] {split.name}: labels={stats['labels_total']}, "
                f"prepared={stats['prepared']}, skipped={stats['skipped']}"
            )
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}")
        return 1

    print("\n=== Summary ===")
    print(f"out_root : {out_root}")
    print(f"labels   : {total_labels}")
    print(f"prepared : {total_prepared}")
    print(f"skipped  : {total_skipped}")
    print(f"mode     : {args.link_mode}")
    print(f"rebuild  : {args.force_rebuild}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

