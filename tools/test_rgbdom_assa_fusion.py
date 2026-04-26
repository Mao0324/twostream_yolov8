"""Smoke tests for RGBDominantASSAFusion."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import RGBDominantASSAFusion
from ultralytics.nn.tasks import OBBModel


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def test_module_forward():
    torch.manual_seed(0)
    shapes = ((2, 128, 80, 80), (2, 256, 40, 40), (2, 512, 20, 20))
    total = 0

    for shape in shapes:
        _, c, _, _ = shape
        module = RGBDominantASSAFusion(c, alpha_init=0.2)
        x_rgb = torch.randn(shape)
        x_ir = torch.randn(shape)

        out = module([x_rgb, x_ir])
        expected = x_rgb + 0.2 * x_ir
        max_abs_diff = (out - expected).abs().max().item()
        params = count_params(module)
        total += params

        assert out.shape == x_rgb.shape, f"shape mismatch for C={c}: {out.shape} != {x_rgb.shape}"
        assert max_abs_diff < 1e-6, f"init mismatch for C={c}: max_abs_diff={max_abs_diff}"
        print(f"C={c}: params={params}, out_shape={tuple(out.shape)}, init_max_abs_diff={max_abs_diff:.3e}")

    print(f"RGBDominantASSAFusion total params for P3/P4/P5: {total}")


def summarize_obb_output(out):
    if isinstance(out, tuple):
        return [summarize_obb_output(x) for x in out]
    if isinstance(out, list):
        return [tuple(x.shape) for x in out]
    return tuple(out.shape)


def test_model_build_and_forward():
    for cfg in ("yaml/yolov8_twostream_obb_rgbdom_assa.yaml", "yaml/PC2f_MPF_yolov8s.yaml"):
        print(f"\nBuilding {cfg}")
        model = OBBModel(cfg, verbose=True)
        model.eval()
        model.info(verbose=True, imgsz=64)
        with torch.no_grad():
            out = model(torch.randn(1, 6, 64, 64))
        print(f"{cfg} forward output: {summarize_obb_output(out)}")


if __name__ == "__main__":
    test_module_forward()
    test_model_build_and_forward()
