# 测试
from ultralytics import YOLO 
model = YOLO('/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/escvehicle_runs/train/weights/best.pt') 
metrics = model.val(data='/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/data/escvehicle.yaml',split='test',imgsz=704,batch=16)