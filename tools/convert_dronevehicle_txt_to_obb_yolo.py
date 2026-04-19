#!/usr/bin/env python3
"""
将 DroneVehicle_SMFDet2 的 OBB 文本标注转换为 YOLO-OBB 归一化格式。

输入行示例（原始）：
154 159 156 221 179 218 175 158 CAR 0

输出行示例（YOLO-OBB）：
0 0.240625 0.310547 0.243750 0.431641 0.279688 0.425781 0.273438 0.308594

转换规则：
1. 每行前 8 个数字视为 4 个顶点坐标：x1 y1 x2 y2 x3 y3 x4 y4（像素坐标）。
2. 第 9 列视为类别名（例如 CAR/TRUCK/...）。
3. 第 10 列及之后内容忽略（例如 difficult/truncated 标记）。
4. 输出为：class_id + 8 个归一化坐标。
5. 默认会将归一化坐标裁剪到 [0, 1]，用于处理原始标注中的负数/越界值。

注意：
- 归一化需要图像宽高。默认使用 width=640, height=512（DroneVehicle 常见尺寸）。
- 如果你的数据尺寸不同，请用 --img-width / --img-height 覆盖。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


# 默认输入目录（按用户要求）
DEFAULT_TRAIN_DIR = "/home/biiteam/Storage-4T/biiteam/WBY/datasets/DroneVehicle_SMFDet2/train_labels"
DEFAULT_TEST_DIR = "/home/biiteam/Storage-4T/biiteam/WBY/datasets/DroneVehicle_SMFDet2/test_labels"
DEFAULT_VAL_DIR = "/home/biiteam/Storage-4T/biiteam/WBY/datasets/DroneVehicle_SMFDet2/val_labels"

# 默认输出根目录（按用户要求）
DEFAULT_OUT_ROOT = "/home/biiteam/Storage-4T/biiteam/MCONG/datasets/dronevehiclelabels_obb"


def build_label_map() -> Dict[str, int]:
    """构建类别名到 class_id 的映射（大小写不敏感，兼容常见写法）。"""
    # 主映射（5 类）
    base = {
        "CAR": 0,
        "TRUCK": 1,
        "BUS": 2,
        "VAN": 3,
        "FREIGHT_CAR": 4,
    }

    # 扩展同义写法
    aliases = {
        "FREIGHTCAR": "FREIGHT_CAR",
        "FREIGHT-CAR": "FREIGHT_CAR",
        "FREIGHT": "FREIGHT_CAR",
        "FERIGHT_CAR": "FREIGHT_CAR",
    }

    mapping: Dict[str, int] = {}
    for k, v in base.items():
        mapping[k] = v
    for src, dst in aliases.items():
        mapping[src] = base[dst]

    return mapping


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Convert DroneVehicle txt labels to YOLO-OBB normalized txt labels")

    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR, help="训练集标签目录")
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR, help="测试集标签目录")
    parser.add_argument("--val-dir", default=DEFAULT_VAL_DIR, help="验证集标签目录")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="输出根目录")

    parser.add_argument("--img-width", type=float, default=640.0, help="图像宽度（用于 x 归一化）")
    parser.add_argument("--img-height", type=float, default=512.0, help="图像高度（用于 y 归一化）")

    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="不裁剪归一化坐标到 [0,1]；默认会裁剪",
    )

    parser.add_argument(
        "--skip-unknown-class",
        action="store_true",
        help="遇到未知类别时跳过该行；默认遇到未知类别会报错终止",
    )

    return parser.parse_args()


def normalize_and_maybe_clip(
    coords: List[float],
    img_w: float,
    img_h: float,
    clip: bool,
) -> List[float]:
    """将 8 个像素坐标归一化，并按需裁剪到 [0,1]。"""
    out: List[float] = []
    for i, v in enumerate(coords):
        if i % 2 == 0:
            n = v / img_w  # x
        else:
            n = v / img_h  # y

        if clip:
            if n < 0.0:
                n = 0.0
            elif n > 1.0:
                n = 1.0

        out.append(n)
    return out


def parse_one_line(
    line: str,
    label_map: Dict[str, int],
    img_w: float,
    img_h: float,
    clip: bool,
    skip_unknown_class: bool,
) -> Tuple[str | None, str | None]:
    """
    解析并转换单行标注。

    返回：
    - (converted_line, None) 表示成功。
    - (None, error_message) 表示失败（或被跳过）。
    """
    s = line.strip()
    if not s:
        return None, None

    parts = s.split()
    if len(parts) < 9:
        return None, f"invalid format (need >=9 columns): {s}"

    # 前 8 列必须是数值坐标
    coords: List[float] = []
    try:
        for i in range(8):
            coords.append(float(parts[i]))
    except ValueError:
        return None, f"invalid coordinate values: {s}"

    # 第 9 列是类别名
    cls_name = parts[8].upper().replace("-", "_")

    # 特殊兼容：若类别里包含空格（极少见），这里无法从 split 可靠还原，
    # 默认按第 9 列解析。常见数据通常是 CAR/TRUCK/BUS/VAN/FREIGHT_CAR。
    if cls_name not in label_map:
        msg = f"unknown class '{parts[8]}'"
        if skip_unknown_class:
            return None, msg
        return None, msg

    cls_id = label_map[cls_name]

    ncoords = normalize_and_maybe_clip(coords, img_w=img_w, img_h=img_h, clip=clip)

    # 输出保留 6 位小数
    out = f"{cls_id} " + " ".join(f"{v:.6f}" for v in ncoords)
    return out, None


def convert_dir(
    in_dir: Path,
    out_dir: Path,
    label_map: Dict[str, int],
    img_w: float,
    img_h: float,
    clip: bool,
    skip_unknown_class: bool,
) -> Tuple[int, int, int]:
    """
    转换一个 split 目录下的所有 txt。

    返回：
    (file_count, line_ok_count, line_skip_or_err_count)
    """
    if not in_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    ok_lines = 0
    bad_lines = 0

    txt_files = sorted(in_dir.glob("*.txt"))
    for txt_path in txt_files:
        file_count += 1

        out_lines: List[str] = []
        lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        for li, line in enumerate(lines, start=1):
            converted, err = parse_one_line(
                line=line,
                label_map=label_map,
                img_w=img_w,
                img_h=img_h,
                clip=clip,
                skip_unknown_class=skip_unknown_class,
            )

            if converted is not None:
                out_lines.append(converted)
                ok_lines += 1
            elif err is None:
                # 空行
                continue
            else:
                bad_lines += 1
                if not skip_unknown_class:
                    raise ValueError(f"{txt_path}:{li}: {err}")
                print(f"[WARN] {txt_path}:{li}: {err}")

        out_path = out_dir / txt_path.name
        out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    return file_count, ok_lines, bad_lines


def main() -> int:
    args = parse_args()

    if args.img_width <= 0 or args.img_height <= 0:
        print("[ERROR] --img-width and --img-height must be positive")
        return 2

    label_map = build_label_map()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    val_dir = Path(args.val_dir)
    out_root = Path(args.out_root)

    # 为避免 train/test/val 同名文件互相覆盖，输出按 split 分子目录保存。
    out_train = out_root / "train_labels"
    out_test = out_root / "test_labels"
    out_val = out_root / "val_labels"

    clip = not args.no_clip

    try:
        t_files, t_ok, t_bad = convert_dir(
            in_dir=train_dir,
            out_dir=out_train,
            label_map=label_map,
            img_w=args.img_width,
            img_h=args.img_height,
            clip=clip,
            skip_unknown_class=args.skip_unknown_class,
        )

        s_files, s_ok, s_bad = convert_dir(
            in_dir=test_dir,
            out_dir=out_test,
            label_map=label_map,
            img_w=args.img_width,
            img_h=args.img_height,
            clip=clip,
            skip_unknown_class=args.skip_unknown_class,
        )

        v_files, v_ok, v_bad = convert_dir(
            in_dir=val_dir,
            out_dir=out_val,
            label_map=label_map,
            img_w=args.img_width,
            img_h=args.img_height,
            clip=clip,
            skip_unknown_class=args.skip_unknown_class,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}")
        return 1

    total_files = t_files + s_files + v_files
    total_ok = t_ok + s_ok + v_ok
    total_bad = t_bad + s_bad + v_bad

    print("=== Conversion Finished ===")
    print(f"train: files={t_files}, converted_lines={t_ok}, skipped_or_bad_lines={t_bad}")
    print(f" test: files={s_files}, converted_lines={s_ok}, skipped_or_bad_lines={s_bad}")
    print(f"  val: files={v_files}, converted_lines={v_ok}, skipped_or_bad_lines={v_bad}")
    print(f"total: files={total_files}, converted_lines={total_ok}, skipped_or_bad_lines={total_bad}")
    print(f"output root: {out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
