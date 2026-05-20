#!/usr/bin/env python3
"""Inspect selected tensor values in a YOLO checkpoint.

Examples:
    python tools/inspect_checkpoint_params.py runs/train/weights/best.pt
    python tools/inspect_checkpoint_params.py runs/train/weights/best.pt scale_rgb scale_ir
    python tools/inspect_checkpoint_params.py runs/train/weights/best.pt model.8.scale_rgb --exact
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
warnings.filterwarnings("ignore", category=FutureWarning)


def install_checkpoint_compat_aliases() -> None:
    """Register lightweight aliases needed by older local checkpoints."""
    try:
        from ultralytics.nn.modules import block
    except Exception:
        return

    if hasattr(block, "ASSARIFusion") and not hasattr(block, "ASSAFusion"):
        block.ASSAFusion = block.ASSARIFusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print checkpoint state_dict entries whose names match one or more patterns."
    )
    parser.add_argument("weights", type=Path, help="Path to a .pt checkpoint, for example runs/.../weights/best.pt")
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["scale_rgb", "scale_ir"],
        help="Key patterns to match. Default: scale_rgb scale_ir",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "ema", "model", "checkpoint"),
        default="auto",
        help="Where to read tensors from. auto prefers ema, then model, then raw checkpoint.",
    )
    parser.add_argument("--exact", action="store_true", help="Require an exact key match instead of substring match.")
    parser.add_argument("--regex", action="store_true", help="Treat patterns as regular expressions.")
    parser.add_argument("--ignore-case", action="store_true", help="Case-insensitive matching.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of matched keys to print.")
    parser.add_argument("--precision", type=int, default=8, help="Floating point print precision.")
    return parser.parse_args()


def torch_load(path: Path):
    install_checkpoint_compat_aliases()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def choose_source(ckpt, source: str):
    if source == "checkpoint":
        return ckpt, "checkpoint"

    if isinstance(ckpt, dict):
        if source in {"auto", "ema"} and ckpt.get("ema") is not None:
            return ckpt["ema"], "ema"
        if source in {"auto", "model"} and ckpt.get("model") is not None:
            return ckpt["model"], "model"
        if source != "auto":
            raise KeyError(f"checkpoint has no usable '{source}' entry")

    return ckpt, "checkpoint"


def to_state_dict(obj):
    if hasattr(obj, "state_dict"):
        return obj.float().state_dict() if hasattr(obj, "float") else obj.state_dict()
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return {k: v for k, v in obj.items() if torch.is_tensor(v)}
    raise TypeError(f"cannot extract state_dict from object of type {type(obj).__name__}")


def key_matches(key: str, patterns: Iterable[str], exact: bool, regex: bool, ignore_case: bool) -> bool:
    flags = re.IGNORECASE if ignore_case else 0
    haystack = key.lower() if ignore_case and not regex else key

    for pattern in patterns:
        if regex:
            if re.search(pattern, key, flags=flags):
                return True
        elif exact:
            needle = pattern.lower() if ignore_case else pattern
            if haystack == needle:
                return True
        else:
            needle = pattern.lower() if ignore_case else pattern
            if needle in haystack:
                return True
    return False


def format_tensor(tensor: torch.Tensor, precision: int) -> str:
    tensor = tensor.detach().cpu()
    if tensor.numel() == 1:
        value = tensor.reshape(-1)[0].item()
        if isinstance(value, float):
            return f"{value:.{precision}g}"
        return str(value)

    data = tensor.float()
    return (
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"mean={data.mean().item():.{precision}g}, "
        f"std={data.std(unbiased=False).item():.{precision}g}, "
        f"min={data.min().item():.{precision}g}, "
        f"max={data.max().item():.{precision}g}"
    )


def main() -> None:
    args = parse_args()
    ckpt = torch_load(args.weights)
    obj, source_name = choose_source(ckpt, args.source)
    state_dict = to_state_dict(obj)

    matches = [
        (key, value)
        for key, value in state_dict.items()
        if torch.is_tensor(value) and key_matches(key, args.patterns, args.exact, args.regex, args.ignore_case)
    ]

    print(f"weights: {args.weights}")
    print(f"source: {source_name}")
    print(f"patterns: {', '.join(args.patterns)}")
    print(f"matches: {len(matches)}")

    if not matches:
        print("No matching tensor keys found.")
        return

    for key, value in matches[: args.limit]:
        print(f"{key}: {format_tensor(value, args.precision)}")

    if len(matches) > args.limit:
        print(f"... skipped {len(matches) - args.limit} more matches; increase --limit to print them.")


if __name__ == "__main__":
    main()
