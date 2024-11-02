import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

results = model.train(data = "config.yaml", epochs=1)


# from ultralytics import YOLO

# model = YOLO("yolov8n.yaml")
# results = model.train(data="./config.yaml", epochs=1)