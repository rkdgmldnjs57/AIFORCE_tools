#!pip install ultralytics comet-ml

# Roboflow download

# import comet_ml
# from comet_ml import Experiment
from ultralytics import YOLO, RTDETR

def train():
    # # Comet
    project_name = "no_more"
    experiment_name = "train_v1"
    # comet_ml.login(project_name=project_name)
    # experiment = Experiment(api_key="mxaExOvwSuIZHKt4MpwmZ2mr0", project_name=project_name, )
    # experiment.set_name(experiment_name)

    # Load a model
    # model = RTDETR("rtdetr-l.pt")
    model = YOLO("yolo11n.pt")

    # Train the model
    results = model.train(
        data="C:/Users/Admin/AIFORCE_tools/final/data.yaml",
        project=project_name,
        name=experiment_name,
        batch=1,
        save_period=1,
        save_json=True,
        epochs=5,
        device=0
    )

if __name__ == '__main__':
    train()