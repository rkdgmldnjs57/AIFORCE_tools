from ultralytics import YOLO, RTDETR

# Load a pretrained YOLO11n model
model = YOLO("yolov8n.pt")
# model = RTDETR("rtdetr-l.engine")

# Define remote image or video URL
source = "real_videos_2/vlog_1930.avi"

# Run inference on the source
results = model.predict(
    source,
    save=True,
    half=False,
    device=0
)  # list of Results objects
