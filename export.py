from ultralytics import YOLO, RTDETR

# Load a model
model = RTDETR("rtdetr-l.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(format="engine", half=False, batch=1, device=0, opset=16)
