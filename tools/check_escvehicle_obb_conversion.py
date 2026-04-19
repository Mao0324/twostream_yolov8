#!/usr/bin/env python3
"""Check JSON-to-OBB TXT conversion correctness for ESCVehicle labels.
用于检查/home/biiteam/Storage-4T/biiteam/MCONG/datasets/ESCVehicle/visible/labels文件夹里面标注文件转换是否正确

Example:
  python tools/check_escvehicle_obb_conversion.py \
    --json-dir /path/to/json_labels \
    --txt-dir /path/to/txt_labels \
    --label-map "car:0"
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Obj:
    cls: Optional[int]
    pts: List[float]  # [x1,y1,x2,y2,x3,y3,x4,y4], normalized


@dataclass
class ParsedTxt:
    cls: int
    pts: List[float]
    raw: str


def parse_label_map(s: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    s = s.strip()
    if not s:
        return mapping
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --label-map item: {item!r}, expected like car:0")
        k, v = item.split(":", 1)
        mapping[k.strip()] = int(v.strip())
    return mapping


def load_json_objs(json_path: Path, label_map: Dict[str, int], strict_label: bool) -> Tuple[List[Obj], List[str]]:
    errors: List[str] = []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [], [f"JSON parse error: {e}"]

    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)) or w <= 0 or h <= 0:
        return [], ["Missing/invalid imageWidth or imageHeight"]

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return [], ["Missing/invalid shapes"]

    objs: List[Obj] = []
    for i, sh in enumerate(shapes):
        if not isinstance(sh, dict):
            errors.append(f"shapes[{i}] is not an object")
            continue

        st = sh.get("shape_type")
        if st != "polygon":
            continue

        label = sh.get("label")
        pts = sh.get("points")

        if not isinstance(label, str):
            errors.append(f"shapes[{i}] missing string label")
            continue
        if not isinstance(pts, list) or len(pts) != 4:
            errors.append(f"shapes[{i}] label={label!r} is not 4-point polygon")
            continue

        flat: List[float] = []
        ok = True
        for j, p in enumerate(pts):
            if (
                not isinstance(p, list)
                or len(p) != 2
                or not isinstance(p[0], (int, float))
                or not isinstance(p[1], (int, float))
            ):
                errors.append(f"shapes[{i}] point[{j}] invalid")
                ok = False
                break
            x = float(p[0]) / float(w)
            y = float(p[1]) / float(h)
            flat.extend([x, y])
        if not ok:
            continue

        if label in label_map:
            cls = label_map[label]
        else:
            cls = None
            msg = f"shapes[{i}] label={label!r} not in --label-map"
            if strict_label:
                errors.append(msg)
                continue
            errors.append(msg + " (class-id check skipped for this object)")

        objs.append(Obj(cls=cls, pts=flat))

    return objs, errors


def load_txt_objs(txt_path: Path) -> Tuple[List[ParsedTxt], List[str]]:
    errors: List[str] = []
    if not txt_path.exists():
        return [], ["Missing TXT file"]

    out: List[ParsedTxt] = []
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    for li, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 9:
            errors.append(f"Line {li}: expected 9 columns, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"Line {li}: non-numeric value")
            continue
        out.append(ParsedTxt(cls=cls, pts=coords, raw=s))
    return out, errors


def point_orders(coords8: Sequence[float]) -> List[List[float]]:
    # Return all equivalent 4-point orders: cyclic rotations + reversed cyclic rotations.
    pts = [(coords8[2 * i], coords8[2 * i + 1]) for i in range(4)]
    variants: List[List[float]] = []

    for r in range(4):
        cyc = pts[r:] + pts[:r]
        variants.append([v for p in cyc for v in p])

    rev = list(reversed(pts))
    for r in range(4):
        cyc = rev[r:] + rev[:r]
        variants.append([v for p in cyc for v in p])

    return variants


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def best_geom_diff(expected8: Sequence[float], got8: Sequence[float]) -> float:
    return min(max_abs_diff(expected8, var) for var in point_orders(got8))


def check_one_file(
    json_path: Path,
    txt_path: Path,
    label_map: Dict[str, int],
    tol: float,
    strict_label: bool,
) -> Tuple[bool, List[str]]:
    ok = True
    msgs: List[str] = []

    expected, e1 = load_json_objs(json_path, label_map, strict_label=strict_label)
    got, e2 = load_txt_objs(txt_path)

    if e1:
        ok = False
        msgs.extend([f"{json_path.name}: {m}" for m in e1])
    if e2:
        ok = False
        msgs.extend([f"{txt_path.name}: {m}" for m in e2])

    if not expected and not got:
        return ok, msgs

    used = [False] * len(got)

    for ei, exp in enumerate(expected):
        best_j = -1
        best_d = math.inf

        for j, item in enumerate(got):
            if used[j]:
                continue
            if exp.cls is not None and item.cls != exp.cls:
                continue
            d = best_geom_diff(exp.pts, item.pts)
            if d < best_d:
                best_d = d
                best_j = j

        if best_j == -1:
            ok = False
            msgs.append(f"{json_path.name}: object[{ei}] cannot find matching TXT line")
            continue

        if best_d > tol:
            ok = False
            msgs.append(
                f"{json_path.name}: object[{ei}] best geom diff={best_d:.8f} > tol={tol:.8f}; "
                f"matched line='{got[best_j].raw}'"
            )
            used[best_j] = True
        else:
            used[best_j] = True

    for j, u in enumerate(used):
        if not u:
            ok = False
            msgs.append(f"{txt_path.name}: extra unmatched line: '{got[j].raw}'")

    return ok, msgs


def collect_json_files(json_dir: Path) -> List[Path]:
    return sorted([p for p in json_dir.glob("*.json") if p.is_file()])


def main() -> int:
    default_dir = "/home/biiteam/Storage-4T/biiteam/MCONG/datasets/ESCVehicle/visible/labels"

    parser = argparse.ArgumentParser(description="Check JSON-to-OBB TXT conversion correctness")
    parser.add_argument("--json-dir", type=str, default=default_dir, help="Directory containing *.json")
    parser.add_argument(
        "--txt-dir",
        type=str,
        default=default_dir,
        help="Directory containing converted *.txt (same stem as json)",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=(
            "car:0,truck:1,bus:2,van:3,freight car:4,suv:5,construction vehicle:6"
        ),
        help="Label to class-id map, e.g. 'car:0,bus:1,truck:2'. Unknown labels skip class check unless --strict-label.",
    )
    parser.add_argument(
        "--strict-label",
        action="store_true",
        help="Treat unknown labels (not in --label-map) as errors.",
    )
    parser.add_argument("--tol", type=float, default=1e-6, help="Max abs diff tolerance for normalized coords")
    parser.add_argument(
        "--stop-on-first-fail",
        action="store_true",
        help="Stop immediately after first failing file.",
    )

    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    txt_dir = Path(args.txt_dir)

    if not json_dir.exists() or not json_dir.is_dir():
        print(f"[ERROR] Invalid --json-dir: {json_dir}")
        return 2
    if not txt_dir.exists() or not txt_dir.is_dir():
        print(f"[ERROR] Invalid --txt-dir: {txt_dir}")
        return 2

    try:
        label_map = parse_label_map(args.label_map)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 2

    json_files = collect_json_files(json_dir)
    if not json_files:
        print(f"[WARN] No json files found in: {json_dir}")
        return 0

    total = len(json_files)
    passed = 0
    failed = 0

    for jp in json_files:
        tp = txt_dir / (jp.stem + ".txt")
        ok, msgs = check_one_file(
            json_path=jp,
            txt_path=tp,
            label_map=label_map,
            tol=args.tol,
            strict_label=args.strict_label,
        )

        if ok:
            passed += 1
        else:
            failed += 1
            print(f"\n[FAIL] {jp.name}")
            for m in msgs:
                print(f"  - {m}")
            if args.stop_on_first_fail:
                break

    print("\n=== Summary ===")
    print(f"JSON files checked : {passed + failed}/{total}")
    print(f"Passed             : {passed}")
    print(f"Failed             : {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
