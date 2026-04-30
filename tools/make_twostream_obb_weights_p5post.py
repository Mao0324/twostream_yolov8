#!/usr/bin/env python3
"""Create weights for yolov8_twostream_obb_assarifusion_p5post.yaml.

This wrapper keeps the original conversion implementation untouched and only
updates the target YAML/default output plus layer-index maps for the extra
post-P5 ASSARIFusion/Silence/ADD layers.
"""

from __future__ import annotations

from pathlib import Path

import make_twostream_obb_weights as base


REPO_ROOT = Path(__file__).resolve().parents[1]

base.DEFAULT_TARGET_YAML = REPO_ROOT / "yaml" / "yolov8_twostream_obb_assarifusion_p5post.yaml"
base.DEFAULT_OUTPUT = REPO_ROOT / "pre-trained" / "yolov8s-obb_twostream_assarifusion_p5post.pt"

# single-stream yolov8s(-obb) layer index -> p5post two-stream RGB/shared layer index
base.SINGLE_TO_TWOSTREAM_RGB = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 9,
    5: 10,
    6: 14,
    7: 15,
    8: 19,
    9: 20,
    12: 31,
    15: 34,
    16: 35,
    18: 37,
    19: 38,
    21: 40,
    22: 41,
}

# p5post two-stream RGB branch layer index -> p5post two-stream IR branch layer index
base.TWOSTREAM_RGB_TO_IR = {
    0: 4,
    1: 5,
    2: 6,
    3: 7,
    9: 11,
    10: 12,
    14: 16,
    15: 17,
    19: 21,
    20: 22,
}


if __name__ == "__main__":
    raise SystemExit(base.main())
