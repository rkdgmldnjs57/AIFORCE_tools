from ultralytics import YOLO, RTDETR

# Load a model
model = RTDETR("path/to/best.pt")

# Validate the model
metrics = model.val(
    data="path/to/data.yaml",
    project="./test_output/",
    name="Dummy",
)
