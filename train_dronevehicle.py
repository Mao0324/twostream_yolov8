# 训练（DroneVehicle）
from ultralytics import YOLO
import ultralytics.nn.tasks  # noqa: F401

# 1) 模型结构
model = YOLO('/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/yaml/yolov8_twostream_obb_mbnet.yaml')

# 2) 预训练权重（如不存在可注释掉）
model.load('/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream_mbnet.pt')

# 3) 训练
results = model.train(
    data='/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/data/dronevehicle.yaml',
    batch=64,
    epochs=100,
    imgsz=640,
    device='5,7',
    project='dronevehicle_runs_mbnet_dense_dmaf_iafa',
    task='obb'
)
