# 训练（DroneVehicle）
from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401

# 1) 模型结构
model = YOLO('/media/biiteam/新加卷/biiteam/MCONG/TwoStream_Yolov8_2/yaml/PC2f_MPF_yolov8s.yaml', task='obb')

# 2) 预训练权重（如不存在可注释掉）
model.load('/media/biiteam/新加卷/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_pc2f_mpf.pt')

# 3) 单卡训练
results = model.train(
    data="/media/biiteam/新加卷/biiteam/MCONG/TwoStream_Yolov8_2/data/dronevehicle.yaml",
    batch=64,
    epochs=100,
    imgsz=640,
    device="2,3",          # 改成实际可用的那张卡
    workers=8,           # 15 GiB RAM: avoid 8 train + 16 val DataLoader workers exhausting host memory
    project="dronevehicle_runs_baseline",
    task="obb",
)
