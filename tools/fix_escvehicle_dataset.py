#!/usr/bin/env python3
"""One-shot fixer for ESCVehicle two-stream training issues.

What it does:
1) Patch ultralytics/data/base.py to read IR images from self.imir_files[i]
   instead of replacing 'images' -> 'image' in RGB paths.
2) Convert Labelme JSON labels to YOLO OBB TXT labels for visible/infrared.
3) Remove stale labels.cache files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/home/biiteam/Storage-4T/biiteam/MCONG/datasets/ESCVehicle")
BASE_PY_REL = Path("/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/ultralytics/data/base.py")


CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "van": 3,
    "freight car": 4,
    "freight_car": 4,
    "suv": 5,
    "construction vehicle": 6,
    "construction_vehicle": 6,
}


def patch_base_py(repo_root: Path) -> bool:
    base_py = repo_root / BASE_PY_REL
    if not base_py.exists():
        raise FileNotFoundError(f"base.py not found: {base_py}")

    text = base_py.read_text(encoding="utf-8")
    updated = text

    # Fix wrong IR path composition.
    wrong_block = re.compile(
        r"f1\s*=\s*self\.im_files\[i\]\.replace\('images','image'\)\s*\n\s*imir\s*=\s*cv2\.imread\(f1\)"
    )
    updated = wrong_block.sub("f1 = self.imir_files[i]\n        imir = cv2.imread(f1)", updated, count=1)

    # Ensure IR read failure is explicit before dstack.
    if "IR Image Not Found" not in updated:
        updated = updated.replace(
            "im = np.dstack((im, imir))",
            'if imir is None:\n                raise FileNotFoundError(f"IR Image Not Found {f1}")\n            im = np.dstack((im, imir))',
            1,
        )

    if updated != text:
        base_py.write_text(updated, encoding="utf-8")
        print(f"[OK] patched {base_py}")
        return True
    else:
        print(f"[SKIP] {base_py} already patched")
        return False


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def to_yolo_obb(points: list[list[float]], w: float, h: float):
    """Convert 4 polygon points to YOLO-OBB normalized format.

    Returns:
        list[float] | None: [x1, y1, x2, y2, x3, y3, x4, y4] (normalized), or None if invalid.
    """
    if w <= 0.0 or h <= 0.0:
        return None
    if len(points) != 4:
        # For this dataset, objects are annotated as 4-point polygons.
        return None

    out = []
    for p in points:
        if len(p) < 2:
            return None
        x = _clip(float(p[0]), 0.0, float(w))
        y = _clip(float(p[1]), 0.0, float(h))
        out.extend([x / w, y / h])
    return out


def convert_labelme_json_to_txt(labels_dir: Path) -> tuple[int, int, int, int, int]:
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")

    n_json = 0
    n_txt = 0
    n_obj_ok = 0
    n_obj_skip = 0
    n_json_parse_fail = 0

    for jf in sorted(labels_dir.glob("*.json")):
        n_json += 1
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            n_json_parse_fail += 1
            continue

        w = float(data.get("imageWidth", 704))
        h = float(data.get("imageHeight", 704))
        out_lines: list[str] = []

        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip().lower()
            cls_id = CLASS_MAP.get(label)
            if cls_id is None:
                n_obj_skip += 1
                continue

            pts = shape.get("points") or []
            if len(pts) < 2:
                n_obj_skip += 1
                continue

            obb = to_yolo_obb(pts, w, h)
            if obb is None:
                n_obj_skip += 1
                continue

            out_lines.append(
                f"{cls_id} "
                f"{obb[0]:.6f} {obb[1]:.6f} "
                f"{obb[2]:.6f} {obb[3]:.6f} "
                f"{obb[4]:.6f} {obb[5]:.6f} "
                f"{obb[6]:.6f} {obb[7]:.6f}"
            )
            n_obj_ok += 1

        jf.with_suffix(".txt").write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""),
            encoding="utf-8",
        )
        n_txt += 1

    return n_json, n_txt, n_obj_ok, n_obj_skip, n_json_parse_fail


def delete_label_cache(dataset_root: Path) -> tuple[int, int]:
    n_removed = 0
    n_skip = 0
    for cache in (
        dataset_root / "visible" / "labels.cache",
        dataset_root / "infrared" / "labels.cache",
    ):
        if cache.exists():
            cache.unlink()
            print(f"[OK] removed {cache}")
            n_removed += 1
        else:
            print(f"[SKIP] cache not found: {cache}")
            n_skip += 1
    return n_removed, n_skip


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix ESCVehicle two-stream dataset issues.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--skip-base-patch", action="store_true")
    parser.add_argument("--skip-label-convert", action="store_true")
    args = parser.parse_args()

    base_patch_patched = 0
    base_patch_skipped = 0
    total_json = 0
    total_txt = 0
    total_obj_ok = 0
    total_obj_skip = 0
    total_json_parse_fail = 0
    cache_removed = 0
    cache_skipped = 0

    if not args.skip_base_patch:
        if patch_base_py(args.repo_root):
            base_patch_patched += 1
        else:
            base_patch_skipped += 1

    if not args.skip_label_convert:
        for modality in ("visible", "infrared"):
            labels_dir = args.dataset_root / modality / "labels"
            n_json, n_txt, n_obj_ok, n_obj_skip, n_json_parse_fail = convert_labelme_json_to_txt(labels_dir)
            print(
                f"[OK] {labels_dir}: json={n_json}, txt_written={n_txt}, "
                f"objects_ok={n_obj_ok}, objects_skipped={n_obj_skip}, json_parse_fail={n_json_parse_fail}"
            )
            total_json += n_json
            total_txt += n_txt
            total_obj_ok += n_obj_ok
            total_obj_skip += n_obj_skip
            total_json_parse_fail += n_json_parse_fail
        cache_removed, cache_skipped = delete_label_cache(args.dataset_root)

    print(
        "[SUMMARY] "
        f"base_patch_patched={base_patch_patched}, "
        f"base_patch_skipped={base_patch_skipped}, "
        f"json_total={total_json}, "
        f"txt_written_total={total_txt}, "
        f"objects_ok_total={total_obj_ok}, "
        f"objects_skipped_total={total_obj_skip}, "
        f"json_parse_fail_total={total_json_parse_fail}, "
        f"cache_removed={cache_removed}, "
        f"cache_skipped={cache_skipped}"
    )

    print("[DONE] ESCVehicle dataset fix completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
