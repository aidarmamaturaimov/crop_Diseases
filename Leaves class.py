
import torch
from ultralytics import YOLO

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolo11n-cls.pt").to(device)

model.train(data="/Users/aidarmamaturaimov/Downloads/PlantVillage_FYP_cl", epochs=10, imgsz=640)



