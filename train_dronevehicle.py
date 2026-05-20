# Train DroneVehicle with the two-stream RT-DETR OBB model.
from argparse import ArgumentParser
from pathlib import Path

from ultralytics import RTDETR
import ultralytics.nn.tasks  # noqa: F401


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = ArgumentParser(description="Train DroneVehicle with RT-DETR OBB two-stream model.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "yaml" / "yolov8_twostream_rtdetr_obb_assafusion_postc2f.yaml"),
        help="Model YAML path.",
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "dronevehicle.yaml"),
        help="DroneVehicle dataset YAML path.",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Optional pretrained weights to load. Leave empty to train from scratch.",
    )
    parser.add_argument("--batch", type=int, default=64, help="Batch size. RT-DETR OBB is heavier than YOLO OBB.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--device", default="5,6", help="CUDA device string, e.g. '0' or '0,1'.")
    parser.add_argument(
        "--project",
        default=str(ROOT / "dronevehicle_runs_rtdetr_obb_twostream"),
        help="Output project directory.",
    )
    parser.add_argument("--name", default="train", help="Run name.")
    return parser.parse_args()


def main():
    args = parse_args()

    model = RTDETR(args.model)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=args.data,
        batch=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        task="obb",
    )


if __name__ == "__main__":
    main()
