#!pip install ultralytics comet-ml

# Roboflow download

import comet_ml
from comet_ml import Experiment
from ultralytics import YOLO, RTDETR

# Comet
project_name = "Dummy"
comet_ml.login(project_name=project_name)
experiment = Experiment(project_name=project_name)

# Load a model
model = RTDETR("rtdetr-l.pt")
# model = YOLO("yolo11l.pt")

# Train the model
results = model.train(
    data="./dataset/data.yaml",
    project=project_name,
    name="rtdetr_baseline",
    batch=-1,
    epochs=50,
    device=0,
)
