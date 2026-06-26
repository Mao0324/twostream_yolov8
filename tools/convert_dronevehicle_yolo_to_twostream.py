#!/usr/bin/env python3
"""Crop DroneVehicle RGB/IR pairs and rebuild YOLO-OBB labels.

Supported input layouts:

Raw DroneVehicle (the default):
    <src>/train/trainimg, val/valimg, test/testimg       RGB images
    <src>/train/trainimgr, val/valimgr, test/testimgr   infrared images
    <src>/train/trainlabelr, val/vallabelr, test/testlabelr infrared XML annotations

Prepared YOLO layout:
    <src>/images/{train,val,test}  RGB images (840x712)
    <src>/image/{train,val,test}   infrared images (840x712)
    <src>/labels/{train,val,test}  source labels (see --label-format)

Output layout expected by this repository's two-stream loader:
    <dst>/images/{train,val,test}  RGB images (640x512)
    <dst>/image/{train,val,test}   infrared images (640x512)
    <dst>/labels/{train,val,test}  normalized YOLO-OBB labels

Each output label row is Ultralytics' normalized DOTA/YOLO-OBB format:
    class_id x1 y1 x2 y2 x3 y3 x4 y4

Supported source rows:
    yolo-obb: class_id x1 y1 x2 y2 x3 y3 x4 y4  (normalized)
    yolo-hbb: class_id cx cy width height             (normalized)
    dota:     x1 y1 x2 y2 x3 y3 x4 y4 class [difficult] (pixels)
    xml:      DroneVehicle polygon XML

``auto`` recognizes these three layouts. A YOLO-HBB row can only produce an
axis-aligned four-point box because its original rotation is not available.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SRC_ROOT = Path("/home/ubuntu/MCONG/datasets/DroneVehicle")
DEFAULT_DST_ROOT = Path("/home/ubuntu/MCONG/datasets/DroneVehicle_twostream")
DEFAULT_SPLITS = ("train", "val", "test")
SOURCE_LAYOUTS = ("auto", "raw", "yolo")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "van",
    4: "freight_car",
}
CLASS_IDS = {name: class_id for class_id, name in CLASS_NAMES.items()}
CLASS_ALIASES = {
    "freightcar": "freight_car",
    "freight-car": "freight_car",
    "freight": "freight_car",
    "feright": "freight_car",
    "feright_car": "freight_car",
    "truvk": "truck",
}
LABEL_FORMATS = ("auto", "xml", "yolo-obb", "yolo-hbb", "dota")


@dataclass(frozen=True)
class CropConfig:
    source_width: int
    source_height: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.source_width - self.left - self.right

    @property
    def height(self) -> int:
        return self.source_height - self.top - self.bottom


@dataclass(frozen=True)
class Sample:
    split: str
    stem: str
    rgb: Path
    infrared: Path
    label: Path


@dataclass(frozen=True)
class SampleStats:
    labels_read: int = 0
    labels_written: int = 0
    labels_clipped: int = 0
    labels_dropped: int = 0
    yolo_obb_rows: int = 0
    yolo_hbb_rows: int = 0
    dota_rows: int = 0
    xml_rows: int = 0
    labels_ignored: int = 0

    def __add__(self, other: "SampleStats") -> "SampleStats":
        return SampleStats(
            labels_read=self.labels_read + other.labels_read,
            labels_written=self.labels_written + other.labels_written,
            labels_clipped=self.labels_clipped + other.labels_clipped,
            labels_dropped=self.labels_dropped + other.labels_dropped,
            yolo_obb_rows=self.yolo_obb_rows + other.yolo_obb_rows,
            yolo_hbb_rows=self.yolo_hbb_rows + other.yolo_hbb_rows,
            dota_rows=self.dota_rows + other.dota_rows,
            xml_rows=self.xml_rows + other.xml_rows,
            labels_ignored=self.labels_ignored + other.labels_ignored,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop DroneVehicle white borders and create a paired two-stream YOLO-OBB dataset."
    )
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument(
        "--source-layout",
        choices=SOURCE_LAYOUTS,
        default="auto",
        help="Input directory layout: raw DroneVehicle or prepared YOLO (default: auto).",
    )
    parser.add_argument(
        "--raw-label-modality",
        choices=("ir", "rgb"),
        default="ir",
        help=(
            "Annotation coordinate frame for raw DroneVehicle input: "
            "'*labelr' for infrared or '*label' for RGB (default: ir)."
        ),
    )
    parser.add_argument("--splits", nargs="+", choices=DEFAULT_SPLITS, default=list(DEFAULT_SPLITS))
    parser.add_argument("--source-width", type=int, default=840)
    parser.add_argument("--source-height", type=int, default=712)
    parser.add_argument("--crop-left", type=int, default=100)
    parser.add_argument("--crop-top", type=int, default=100)
    parser.add_argument("--crop-right", type=int, default=100)
    parser.add_argument("--crop-bottom", type=int, default=100)
    parser.add_argument(
        "--label-format",
        choices=LABEL_FORMATS,
        default="auto",
        help=(
            "Source label layout. 'auto' recognizes normalized YOLO-OBB, "
            "normalized YOLO-HBB, pixel-coordinate DOTA rows, and DroneVehicle XML (default: auto)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel image workers (default: up to 8).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG output quality in the range 1-100 (default: 95).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N samples per split; 0 processes all samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing generated split directories before conversion.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only rebuild labels; keep already converted RGB/infrared images.",
    )
    return parser.parse_args()


def index_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate stem '{path.stem}' in {directory}")
        result[path.stem] = path
    return result


def describe_stems(stems: set[str]) -> str:
    preview = ", ".join(sorted(stems)[:5])
    return f"{len(stems)} ({preview}{', ...' if len(stems) > 5 else ''})"


def detect_source_layout(src_root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    yolo_marker = src_root / "images" / "train"
    raw_marker = src_root / "train" / "trainimg"
    if yolo_marker.is_dir():
        return "yolo"
    if raw_marker.is_dir():
        return "raw"
    raise FileNotFoundError(
        f"Cannot recognize source layout under {src_root}. Expected either "
        f"{yolo_marker} or {raw_marker}."
    )


def source_directories(
    src_root: Path,
    split: str,
    source_layout: str,
    raw_label_modality: str,
) -> tuple[Path, Path, Path]:
    if source_layout == "yolo":
        return (
            src_root / "images" / split,
            src_root / "image" / split,
            src_root / "labels" / split,
        )
    if source_layout == "raw":
        # In the official DroneVehicle tree, *img is visible/RGB and *imgr is
        # grayscale infrared. The two modalities have separate annotations.
        label_suffix = f"{split}labelr" if raw_label_modality == "ir" else f"{split}label"
        return (
            src_root / split / f"{split}img",
            src_root / split / f"{split}imgr",
            src_root / split / label_suffix,
        )
    raise AssertionError(f"Unhandled source layout: {source_layout}")


def collect_samples(
    src_root: Path,
    split: str,
    limit: int,
    source_layout: str,
    raw_label_modality: str,
) -> list[Sample]:
    rgb_dir, infrared_dir, label_dir = source_directories(
        src_root, split, source_layout, raw_label_modality
    )
    rgb = index_files(rgb_dir, IMAGE_SUFFIXES)
    infrared = index_files(infrared_dir, IMAGE_SUFFIXES)
    labels = index_files(label_dir, {".xml"} if source_layout == "raw" else {".txt"})

    all_stems = set(rgb) | set(infrared) | set(labels)
    errors: list[str] = []
    for name, files in (("RGB", rgb), ("infrared", infrared), ("labels", labels)):
        missing = all_stems - set(files)
        if missing:
            errors.append(f"missing from {name}: {describe_stems(missing)}")
    if errors:
        raise ValueError(f"[{split}] unpaired dataset files; " + "; ".join(errors))
    if not all_stems:
        raise ValueError(f"[{split}] no paired samples found under {src_root}")

    stems = sorted(all_stems)
    if limit > 0:
        stems = stems[:limit]
    return [Sample(split, stem, rgb[stem], infrared[stem], labels[stem]) for stem in stems]


def polygon_area(coords: list[float]) -> float:
    points = list(zip(coords[0::2], coords[1::2]))
    return abs(
        sum(
            points[i][0] * points[(i + 1) % 4][1]
            - points[(i + 1) % 4][0] * points[i][1]
            for i in range(4)
        )
    ) / 2.0


def parse_class_id(value: str, location: str) -> int:
    """Parse either the required numeric class ID or a DOTA class name."""
    try:
        numeric = float(value)
    except ValueError:
        normalized_name = value.strip().lower().replace(" ", "_")
        normalized_name = CLASS_ALIASES.get(normalized_name, normalized_name)
        if normalized_name not in CLASS_IDS:
            raise ValueError(
                f"{location}: unknown class '{value}'; expected one of {list(CLASS_IDS)}"
            )
        return CLASS_IDS[normalized_name]

    class_id = int(numeric)
    if numeric != class_id or class_id not in CLASS_NAMES:
        raise ValueError(
            f"{location}: class must be one of {sorted(CLASS_NAMES)}, got {value}"
        )
    return class_id


def parse_finite_floats(values: list[str], location: str) -> list[float]:
    try:
        result = [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{location}: non-numeric coordinate value") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{location}: coordinate contains NaN or infinity")
    return result


def detect_label_format(parts: list[str], location: str) -> str:
    """Recognize unambiguous, commonly used DroneVehicle text layouts."""
    if len(parts) == 5:
        return "yolo-hbb"
    if len(parts) == 9:
        try:
            first = float(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            return "dota"
        if first.is_integer() and int(first) in CLASS_NAMES and all(0.0 <= value <= 1.0 for value in coords):
            return "yolo-obb"
        return "dota"
    if len(parts) >= 10:
        return "dota"
    raise ValueError(
        f"{location}: cannot detect label format from {len(parts)} columns; "
        "expected 5 (YOLO-HBB), 9 (YOLO-OBB/DOTA), or >=10 (DOTA)"
    )


def xml_source_rows(
    label_path: Path,
    crop: CropConfig,
) -> tuple[list[tuple[int, list[float], str]], int]:
    """Read RGB-coordinate four-point polygons from a DroneVehicle XML file."""
    try:
        root = ET.parse(label_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{label_path}: malformed XML: {exc}") from exc

    size = root.find("size")
    if size is not None:
        width = size.findtext("width")
        height = size.findtext("height")
        if width is not None and height is not None:
            try:
                xml_size = (int(width), int(height))
            except ValueError as exc:
                raise ValueError(f"{label_path}: invalid XML image size") from exc
            expected_size = (crop.source_width, crop.source_height)
            if xml_size != expected_size:
                raise ValueError(
                    f"{label_path}: XML image size is {xml_size[0]}x{xml_size[1]}, "
                    f"expected {expected_size[0]}x{expected_size[1]}"
                )

    rows: list[tuple[int, list[float], str]] = []
    ignored = 0
    coordinate_names = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
    for object_index, object_node in enumerate(root.findall("object"), 1):
        location = f"{label_path}:object[{object_index}]"
        class_name = object_node.findtext("name")
        if not class_name:
            raise ValueError(f"{location}: missing class name")
        if class_name.strip() == "*":
            ignored += 1
            continue
        polygon = object_node.find("polygon")
        bndbox = object_node.find("bndbox")
        if polygon is not None:
            values = [polygon.findtext(name) for name in coordinate_names]
            if any(value is None for value in values):
                raise ValueError(f"{location}: polygon must contain x1,y1,...,x4,y4")
        elif bndbox is not None:
            xmin = bndbox.findtext("xmin")
            ymin = bndbox.findtext("ymin")
            xmax = bndbox.findtext("xmax")
            ymax = bndbox.findtext("ymax")
            if any(value is None for value in (xmin, ymin, xmax, ymax)):
                raise ValueError(f"{location}: bndbox must contain xmin,ymin,xmax,ymax")
            values = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        else:
            # A few source objects contain only a center point. Their extent
            # and orientation are unknowable, so they cannot form an OBB.
            ignored += 1
            continue
        class_id = parse_class_id(class_name, location)
        pixels = parse_finite_floats([str(value) for value in values], location)
        rows.append((class_id, pixels, "xml"))
    return rows, ignored


def parse_source_row(
    parts: list[str],
    label_format: str,
    crop: CropConfig,
    location: str,
) -> tuple[int, list[float], str]:
    """Return class ID, four source-image pixel points, and detected format."""
    row_format = detect_label_format(parts, location) if label_format == "auto" else label_format

    if row_format == "yolo-obb":
        if len(parts) != 9:
            raise ValueError(f"{location}: YOLO-OBB requires exactly 9 columns, got {len(parts)}")
        class_id = parse_class_id(parts[0], location)
        normalized = parse_finite_floats(parts[1:], location)
        if any(value < 0.0 or value > 1.0 for value in normalized):
            raise ValueError(f"{location}: YOLO-OBB coordinates must be normalized to [0, 1]")
        pixels = [
            value * (crop.source_width if index % 2 == 0 else crop.source_height)
            for index, value in enumerate(normalized)
        ]
        return class_id, pixels, row_format

    if row_format == "yolo-hbb":
        if len(parts) != 5:
            raise ValueError(f"{location}: YOLO-HBB requires exactly 5 columns, got {len(parts)}")
        class_id = parse_class_id(parts[0], location)
        cx, cy, width, height = parse_finite_floats(parts[1:], location)
        if any(value < 0.0 or value > 1.0 for value in (cx, cy, width, height)):
            raise ValueError(f"{location}: YOLO-HBB coordinates must be normalized to [0, 1]")
        x1 = (cx - width / 2.0) * crop.source_width
        x2 = (cx + width / 2.0) * crop.source_width
        y1 = (cy - height / 2.0) * crop.source_height
        y2 = (cy + height / 2.0) * crop.source_height
        return class_id, [x1, y1, x2, y1, x2, y2, x1, y2], row_format

    if row_format == "dota":
        if len(parts) < 9:
            raise ValueError(f"{location}: DOTA requires at least 9 columns, got {len(parts)}")
        pixels = parse_finite_floats(parts[:8], location)
        class_id = parse_class_id(parts[8], location)
        return class_id, pixels, row_format

    raise AssertionError(f"Unhandled label format: {row_format}")


def crop_polygon(coords: list[float], crop: CropConfig) -> tuple[list[float] | None, bool]:
    """Intersect a four-point annotation with the retained image rectangle."""
    points = np.asarray(coords, dtype=np.float32).reshape(4, 2)
    points[:, 0] -= crop.left
    points[:, 1] -= crop.top
    inside = bool(
        np.all((points[:, 0] >= 0.0) & (points[:, 0] <= crop.width))
        and np.all((points[:, 1] >= 0.0) & (points[:, 1] <= crop.height))
    )
    if inside:
        result = points.reshape(-1).astype(float).tolist()
        return (result if polygon_area(result) > 1e-6 else None), False

    crop_box = np.asarray(
        [[0.0, 0.0], [crop.width, 0.0], [crop.width, crop.height], [0.0, crop.height]],
        dtype=np.float32,
    )
    hull = cv2.convexHull(points).reshape(-1, 2)
    area, intersection = cv2.intersectConvexConvex(hull, crop_box)
    if intersection is None or area <= 1e-6:
        return None, True

    # A rectangle clipped at a corner can have more than four vertices. Refit
    # the visible polygon to the four-point OBB layout required by YOLO-OBB.
    visible = intersection.reshape(-1, 2)
    fitted = cv2.boxPoints(cv2.minAreaRect(visible)).astype(np.float64)
    fitted[:, 0] = np.clip(fitted[:, 0], 0.0, crop.width)
    fitted[:, 1] = np.clip(fitted[:, 1], 0.0, crop.height)
    result = fitted.reshape(-1).tolist()
    return (result if polygon_area(result) > 1e-6 else None), True


def convert_label_file(
    label_path: Path,
    crop: CropConfig,
    label_format: str,
) -> tuple[str, SampleStats]:
    output_lines: list[str] = []
    labels_read = 0
    labels_clipped = 0
    labels_dropped = 0
    labels_ignored = 0
    format_counts = {"yolo-obb": 0, "yolo-hbb": 0, "dota": 0, "xml": 0}

    use_xml = label_format == "xml" or (label_format == "auto" and label_path.suffix.lower() == ".xml")
    if use_xml:
        source_rows, labels_ignored = xml_source_rows(label_path, crop)
    else:
        source_rows = []
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            location = f"{label_path}:{line_number}"
            source_rows.append(parse_source_row(line.split(), label_format, crop, location))

    for class_id, source_pixels, row_format in source_rows:
        labels_read += 1
        format_counts[row_format] += 1
        cropped_pixels, was_clipped = crop_polygon(source_pixels, crop)
        if cropped_pixels is None:
            labels_dropped += 1
            continue
        if was_clipped:
            labels_clipped += 1

        output_coords = [
            value / (crop.width if index % 2 == 0 else crop.height)
            for index, value in enumerate(cropped_pixels)
        ]
        output_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in output_coords))

    content = "\n".join(output_lines) + ("\n" if output_lines else "")
    return content, SampleStats(
        labels_read=labels_read,
        labels_written=len(output_lines),
        labels_clipped=labels_clipped,
        labels_dropped=labels_dropped,
        yolo_obb_rows=format_counts["yolo-obb"],
        yolo_hbb_rows=format_counts["yolo-hbb"],
        dota_rows=format_counts["dota"],
        xml_rows=format_counts["xml"],
        labels_ignored=labels_ignored,
    )


def read_and_crop_image(path: Path, crop: CropConfig):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    height, width = image.shape[:2]
    if (width, height) != (crop.source_width, crop.source_height):
        raise ValueError(
            f"Unexpected image size for {path}: {width}x{height}, "
            f"expected {crop.source_width}x{crop.source_height}"
        )
    return image[crop.top : crop.source_height - crop.bottom, crop.left : crop.source_width - crop.right]


def write_image(path: Path, image, jpeg_quality: int) -> None:
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Failed to write image: {path}")


def process_sample(
    sample: Sample,
    dst_root: Path,
    crop: CropConfig,
    jpeg_quality: int,
    label_format: str,
    labels_only: bool,
) -> SampleStats:
    label_content, stats = convert_label_file(sample.label, crop, label_format)
    label_dst = dst_root / "labels" / sample.split / f"{sample.stem}.txt"
    if not labels_only:
        rgb_crop = read_and_crop_image(sample.rgb, crop)
        ir_crop = read_and_crop_image(sample.infrared, crop)
        if rgb_crop.shape[:2] != (crop.height, crop.width):
            raise AssertionError(f"Internal RGB crop size error for {sample.rgb}")
        if ir_crop.shape[:2] != (crop.height, crop.width):
            raise AssertionError(f"Internal infrared crop size error for {sample.infrared}")

        # Both streams use the RGB suffix so consumers that pair exact filenames also work.
        image_name = f"{sample.stem}{sample.rgb.suffix.lower()}"
        rgb_dst = dst_root / "images" / sample.split / image_name
        ir_dst = dst_root / "image" / sample.split / image_name
        write_image(rgb_dst, rgb_crop, jpeg_quality)
        write_image(ir_dst, ir_crop, jpeg_quality)
    label_dst.write_text(label_content, encoding="utf-8")
    return stats


def prepare_output(
    dst_root: Path,
    splits: list[str],
    overwrite: bool,
    labels_only: bool,
) -> None:
    modalities = ("labels",) if labels_only else ("images", "image", "labels")
    generated_dirs = [dst_root / modality / split for modality in modalities for split in splits]
    populated = [directory for directory in generated_dirs if directory.exists() and any(directory.iterdir())]
    if populated and not overwrite:
        raise FileExistsError(
            f"Output already contains files: {populated[0]}. Use --overwrite to rebuild generated splits."
        )
    if overwrite:
        for directory in generated_dirs:
            if directory.exists():
                shutil.rmtree(directory)
    for directory in generated_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    crop = CropConfig(
        source_width=args.source_width,
        source_height=args.source_height,
        left=args.crop_left,
        top=args.crop_top,
        right=args.crop_right,
        bottom=args.crop_bottom,
    )
    if min(crop.source_width, crop.source_height, crop.left, crop.top, crop.right, crop.bottom) < 0:
        raise SystemExit("Image dimensions and crop margins must be non-negative.")
    if crop.width <= 0 or crop.height <= 0:
        raise SystemExit("Crop margins leave an empty output image.")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive.")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100.")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative.")

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    if src_root == dst_root:
        raise SystemExit("Source and destination roots must be different.")

    source_layout = detect_source_layout(src_root, args.source_layout)
    if source_layout == "raw" and args.label_format not in {"auto", "xml"}:
        raise SystemExit("Raw DroneVehicle layout requires --label-format auto or xml.")

    samples_by_split = {
        split: collect_samples(
            src_root, split, args.limit, source_layout, args.raw_label_modality
        )
        for split in args.splits
    }
    prepare_output(dst_root, args.splits, args.overwrite, args.labels_only)

    total_stats = SampleStats()
    total_samples = sum(len(samples) for samples in samples_by_split.values())
    completed = 0
    print(
        f"Converting {total_samples} paired samples from {src_root} to {dst_root} "
        f"using {source_layout} layout "
        f"({crop.source_width}x{crop.source_height} -> {crop.width}x{crop.height})"
        + (" [labels only]" if args.labels_only else "")
    )

    for split in args.splits:
        samples = samples_by_split[split]
        split_stats = SampleStats()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(
                lambda sample: process_sample(
                    sample,
                    dst_root,
                    crop,
                    args.jpeg_quality,
                    args.label_format,
                    args.labels_only,
                ),
                samples,
            )
            for stats in results:
                split_stats += stats
                completed += 1
                if completed % 500 == 0 or completed == total_samples:
                    print(f"Progress: {completed}/{total_samples}", flush=True)
        total_stats += split_stats
        print(
            f"[{split}] images={len(samples)}, labels={split_stats.labels_written}/"
            f"{split_stats.labels_read}, clipped={split_stats.labels_clipped}, "
            f"dropped={split_stats.labels_dropped}, ignored={split_stats.labels_ignored}"
        )

    print(
        "Done: "
        f"images={total_samples}, labels={total_stats.labels_written}/{total_stats.labels_read}, "
        f"clipped={total_stats.labels_clipped}, dropped={total_stats.labels_dropped}, "
        f"ignored={total_stats.labels_ignored}, "
        f"formats=(yolo-obb:{total_stats.yolo_obb_rows}, "
        f"yolo-hbb:{total_stats.yolo_hbb_rows}, dota:{total_stats.dota_rows}, "
        f"xml:{total_stats.xml_rows}), "
        f"output={dst_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
