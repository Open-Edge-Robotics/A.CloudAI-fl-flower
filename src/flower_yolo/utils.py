import os

from ultralytics import YOLO

from flwr.common import NDArrays

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    model = YOLO("yolo11n.pt")
    return model


def get_weights(model: YOLO) -> NDArrays:
    return [val.numpy() for _, val in model.state_dict().items()]
