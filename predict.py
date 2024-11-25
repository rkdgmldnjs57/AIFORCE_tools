from ultralytics import YOLO, RTDETR

# Load a pretrained YOLO11n model
model = RTDETR("rtdetr-l.pt")
# model = RTDETR("rtdetr-l.engine")

# Define remote image or video URL
source = "https://ultralytics.com/images/bus.jpg"

# Run inference on the source
results = model.predict(
    source,
    conf=0.25,
)  # list of Results objects
