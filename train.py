from ultralytics import YOLO

# YOLOv11 
model = YOLO ("yolo11s.pt")  # load a pretrained model (recommended for training) #(r"C:\Users\RYZEN\cctv_mon\project_yolov11_obb\runs\detect\train2\weights\best.pt") #
model.train(
    data="./data.yaml",
    epochs=200,
    imgsz=(640, 480),  
    batch=8,
    workers=0,  
    cache=False, 
    device=0,
    overlap_mask=False  
)