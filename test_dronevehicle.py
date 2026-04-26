from pathlib import Path
import argparse

#python test_dronevehicle.py   --weights /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs_assa_ir_to_rgb/train/weights/best.pt   --project /home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/dronevehicle_runs_assa_ir_to_rgb/train  --name test_results
def parse_args():
    repo_root = Path(__file__).resolve().parent
    default_data = repo_root / "data" / "dronevehicle.yaml"

    parser = argparse.ArgumentParser(description="Test DroneVehicle on test split.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights, e.g. runs/.../weights/best.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help="Dataset yaml path (default: data/dronevehicle.yaml)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device, e.g. 0 or 0,1")
    parser.add_argument("--project", type=str, default=None, help="Output project directory")
    parser.add_argument("--name", type=str, default="test_dronevehicle", help="Run name")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    import ultralytics.nn.tasks  # noqa: F401

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        task="obb",
    )
    print(metrics)


if __name__ == "__main__":
    main()
