from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Use a pre-trained model
model.train(data="data.yaml", epochs=50, device="cpu")  # Use GPU if available

results = model("front90.jpg", save=True, conf=0.2)
