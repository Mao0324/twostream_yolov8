"""Generate Grad-CAM heatmaps for paired RGB/IR test images.

Example:
    python tools/generate_test_pair_heatmaps.py \
        --weight runs/train/weights/best.pt \
        --data data/dronevehicle.yaml \
        --count 5 \
        --layer 20 \
        --output runs/heatmaps/test_pairs

The script accepts paired test sources from a dataset YAML, explicit directories,
or explicit text files. RGB/IR files are paired line-by-line for two text files;
otherwise they are paired by filename stem.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect.obbHeapmap import ActivationsAndGradients, letterbox, yolov8_target  # noqa: E402
from pytorch_grad_cam import EigenCAM, EigenGradCAM, GradCAM, GradCAMPlusPlus, LayerCAM, RandomCAM, XGradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from ultralytics.nn.tasks import attempt_load_weights  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CAM_METHODS = {
    "GradCAM": GradCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "LayerCAM": LayerCAM,
    "RandomCAM": RandomCAM,
    "EigenGradCAM": EigenGradCAM,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", required=True, help="Path to model weight, e.g. best.pt.")
    parser.add_argument("--data", default="data/dronevehicle.yaml", help="Dataset YAML containing test and test_ir.")
    parser.add_argument("--rgb-test", default=None, help="Override RGB test dir or txt file.")
    parser.add_argument("--ir-test", default=None, help="Override IR test dir or txt file.")
    parser.add_argument("--selected-list", default=None, help="Optional txt file with RGB paths to prefer/select from.")
    parser.add_argument("--output", default="runs/heatmaps/test_pairs", help="Output directory.")
    parser.add_argument("--count", type=int, default=5, help="Number of paired images to process.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when sampling pairs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Letterbox size.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--method", default="GradCAM", choices=sorted(CAM_METHODS), help="CAM method.")
    parser.add_argument("--layer", type=int, nargs="+", default=[20], help="Target model layer index/indices.")
    parser.add_argument("--backward-type", default="all", choices=("class", "box", "all"), help="Grad-CAM target type.")
    parser.add_argument("--conf-threshold", type=float, default=0.2, help="Target confidence threshold.")
    parser.add_argument("--ratio", type=float, default=0.02, help="Top prediction ratio used by Grad-CAM target.")
    parser.add_argument("--no-shuffle", action="store_true", help="Take the first N pairs instead of random sampling.")
    return parser.parse_args()


def repo_fallback(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.exists():
        return path

    # YAML files in this repo sometimes contain server absolute paths pointing
    # back to files that also exist under the local repository data directory.
    local_data = ROOT / "data" / path.name
    if local_data.exists():
        return local_data

    return path


def read_yaml_sources(data_yaml: str | Path) -> tuple[Path | None, Path | None]:
    path = repo_fallback(data_yaml)
    if path is None or not path.exists():
        return None, None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = Path(data.get("path", "")) if data.get("path") else None

    def resolve_field(name: str) -> Path | None:
        value = data.get(name)
        if not value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute() and root is not None:
            candidate = root / candidate
        return repo_fallback(candidate)

    return resolve_field("test"), resolve_field("test_ir")


def read_image_list(source: Path) -> list[Path]:
    source = repo_fallback(source)
    if source is None:
        return []
    if source.is_file() and source.suffix.lower() == ".txt":
        with source.open("r", encoding="utf-8") as f:
            return [Path(line.strip()) for line in f if line.strip()]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    return []


def index_by_stem(paths: list[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


def replace_source_dir(path: Path, rgb_source: Path | None, ir_source: Path | None) -> Path | None:
    if ir_source is None or not ir_source.is_dir():
        return None
    if rgb_source is not None and rgb_source.is_dir():
        try:
            rel = path.relative_to(rgb_source)
            direct = ir_source / rel
            if direct.exists():
                return direct
        except ValueError:
            pass

    for suffix in IMAGE_SUFFIXES:
        candidate = ir_source / f"{path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_pairs(rgb_source: Path, ir_source: Path, selected_list: Path | None) -> list[tuple[Path, Path]]:
    rgb_paths = read_image_list(selected_list) if selected_list else read_image_list(rgb_source)
    ir_paths = read_image_list(ir_source)

    if not rgb_paths:
        raise FileNotFoundError(f"No RGB test images found from {rgb_source}.")
    if not ir_paths and not ir_source.is_dir():
        raise FileNotFoundError(f"No IR test images found from {ir_source}.")

    pairs: list[tuple[Path, Path]] = []

    if rgb_source.is_file() and ir_source.is_file() and ir_source.suffix.lower() == ".txt" and not selected_list:
        if len(rgb_paths) != len(ir_paths):
            raise ValueError(f"RGB/IR test lists have different lengths: {len(rgb_paths)} vs {len(ir_paths)}.")
        pairs = list(zip(rgb_paths, ir_paths))
    else:
        ir_by_stem = index_by_stem(ir_paths)
        for rgb_path in rgb_paths:
            ir_path = ir_by_stem.get(rgb_path.stem)
            if ir_path is None:
                ir_path = replace_source_dir(rgb_path, rgb_source if rgb_source.is_dir() else None, ir_source)
            if ir_path is not None:
                pairs.append((rgb_path, ir_path))

    if not pairs:
        raise FileNotFoundError("No paired RGB/IR test images were found. Check --rgb-test and --ir-test.")
    return pairs


def load_cam(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = attempt_load_weights(args.weight, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    target_layers = [model.model[i] for i in args.layer]
    target = yolov8_target(args.backward_type, args.conf_threshold, args.ratio)
    cam_cls = CAM_METHODS[args.method]
    try:
        cam = cam_cls(model, target_layers, use_cuda=device.type == "cuda")
    except TypeError:
        cam = cam_cls(model=model, target_layers=target_layers)
    cam.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
    return model, cam, target, device


def read_pair_tensor(rgb_path: Path, ir_path: Path, imgsz: int, device: torch.device):
    rgb_bgr = cv2.imread(str(rgb_path))
    ir_bgr = cv2.imread(str(ir_path))
    if rgb_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB image: {rgb_path}")
    if ir_bgr is None:
        raise FileNotFoundError(f"Cannot read IR image: {ir_path}")

    rgb = letterbox(rgb_bgr, new_shape=(imgsz, imgsz))[0]
    ir = letterbox(ir_bgr, new_shape=(imgsz, imgsz))[0]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    stacked = np.concatenate((rgb, ir), axis=2)
    tensor = torch.from_numpy(stacked.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return rgb, ir, tensor


def labeled_image(image: np.ndarray, label: str) -> Image.Image:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 24), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return pil


def save_pair_heatmap(
    rgb_path: Path,
    ir_path: Path,
    out_dir: Path,
    index: int,
    cam,
    target,
    imgsz: int,
    device: torch.device,
) -> None:
    rgb, ir, tensor = read_pair_tensor(rgb_path, ir_path, imgsz, device)
    grayscale_cam = cam(tensor, [target])[0]

    rgb_heatmap = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    ir_heatmap = show_cam_on_image(ir, grayscale_cam, use_rgb=True)

    stem = rgb_path.stem
    pair_dir = out_dir / f"{index:02d}_{stem}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(pair_dir / "rgb_letterbox.png")
    Image.fromarray((ir * 255).astype(np.uint8)).save(pair_dir / "ir_letterbox.png")
    Image.fromarray(rgb_heatmap).save(pair_dir / "rgb_heatmap.png")
    Image.fromarray(ir_heatmap).save(pair_dir / "ir_heatmap.png")

    tiles = [
        labeled_image((rgb * 255).astype(np.uint8), "RGB"),
        labeled_image(rgb_heatmap, "RGB heatmap"),
        labeled_image((ir * 255).astype(np.uint8), "IR"),
        labeled_image(ir_heatmap, "IR heatmap"),
    ]
    preview = Image.new("RGB", (tiles[0].width * 4, tiles[0].height), color=(255, 255, 255))
    for i, tile in enumerate(tiles):
        preview.paste(tile, (i * tile.width, 0))
    preview.save(pair_dir / "preview.png")


def main() -> None:
    args = parse_args()
    rgb_source, ir_source = read_yaml_sources(args.data)
    if args.rgb_test:
        rgb_source = repo_fallback(args.rgb_test)
    if args.ir_test:
        ir_source = repo_fallback(args.ir_test)
    selected_list = repo_fallback(args.selected_list) if args.selected_list else None

    if rgb_source is None or ir_source is None:
        raise ValueError("RGB/IR test sources are missing. Provide --data with test/test_ir or pass --rgb-test/--ir-test.")

    pairs = build_pairs(rgb_source, ir_source, selected_list)
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.count]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, cam, target, device = load_cam(args)
    manifest = []
    for idx, (rgb_path, ir_path) in enumerate(pairs, start=1):
        save_pair_heatmap(rgb_path, ir_path, out_dir, idx, cam, target, args.imgsz, device)
        manifest.append(f"{idx:02d}\t{rgb_path}\t{ir_path}")
        print(f"[{idx}/{len(pairs)}] saved heatmaps for {rgb_path.name} + {ir_path.name}")

    (out_dir / "selected_pairs.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Done. Heatmaps saved to: {out_dir}")


if __name__ == "__main__":
    main()
