from ultralytics import YOLO

model = YOLO("yolo11l.pt")
model.predict(source=0, show=True, conf=0.25)