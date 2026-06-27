# 训练（DroneVehicle）
from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401

# 1) 模型结构
model = YOLO('/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/yaml/yolov8_twostream_obb_assafusion_postc2f_iafa_before_fpn_lastdmaf_fasterir_p2c2f.yaml', task='obb')

# 2) 预训练权重（如不存在可注释掉）
model.load('/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream.pt')

# 3) 单卡训练
results = model.train(
    data="/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/data/dronevehicle.yaml",
    batch=64,
    nbs=64,              # 自动累计2次，保持有效batch=64
    lr0=0.01,            # 不需要减半
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    epochs=100,
    imgsz=640,
    device="3,5",          # 改成实际可用的那张卡
    workers=4,           # 15 GiB RAM: avoid 8 train + 16 val DataLoader workers exhausting host memory
    amp=True,
    project="dronevehicle_runs_lastdmaf_fasterir_p2c2f双卡版",
    task="obb",
)
