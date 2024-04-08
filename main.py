from ultralytics import YOLO


# Load model
model = YOLO('yolov8n.pt')

# Inference
result = model(source=0, show=False, stream = True)

# Print results
for r in result:
    boxes = r.boxes.xyxy[0]
    print(f"boxes dimensions {boxes}\n")
