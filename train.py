#训练
from ultralytics import YOLO
import ultralytics.nn.tasks
model = YOLO('/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/yaml/PC2f_MPF_yolov8s.yaml')
model.load("/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/pre-trained/yolov8s-obb_twostream.pt")
results = model.train(
    data='/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/data/escvehicle.yaml',
    batch=64,
    epochs=200,
    imgsz=704,
    device='1,2,3,4',
    project='/home/biiteam/Storage-4T/biiteam/MCONG/TwoStream_Yolov8_2/escvehicle_runs',
    task="obb"
)