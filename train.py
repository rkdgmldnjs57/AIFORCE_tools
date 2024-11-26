#!pip install ultralytics comet-ml

# Roboflow download

import comet_ml
from comet_ml import Experiment
from ultralytics import YOLO, RTDETR

# Comet
project_name = "no_more"
experiment_name = "train_v1"
comet_ml.login(project_name=project_name)
experiment = Experiment(project_name=project_name)
experiment.set_name(experiment_name)

# Load a model
model = RTDETR("rtdetr-l.pt")
# model = YOLO("yolo11l.pt")

# Train the model
results = model.train(
    data="/content/drive/MyDrive/Maicon/datasets/alldata/train_v4.yaml",
    project=project_name,
    name=experiment_name,
    batch=-1,
    save_period=1,
    save_json=True,
    epochs=100,
)
