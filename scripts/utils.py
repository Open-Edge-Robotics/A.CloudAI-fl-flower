import os
import glob
import numpy as np
from logging import INFO
from flwr.common.logger import log
from collections import OrderedDict

import torch

from ultralytics.models import YOLO

from flwr.common import NDArrays

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(model: str = "yolo11x.pt"):
    log(INFO, f"Loading model: {model}")
    return YOLO(model)


def load_model_with_checkpoint(model_name: str = "yolo11x.pt"):
    model = load_model(model_name)

    checkpoint_path: str = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if os.getenv("KUBERNETES_SERVICE_HOST") is not None:
        checkpoint_path = "/mnt/checkpoints/"

    list_of_files = [fname for fname in glob.glob(checkpoint_path + "/*.npz")]

    log(INFO, f"List of files: {list_of_files}")

    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        log(INFO, f"Latest file: {latest_file}")
        checkpoint_file_path = os.path.join(checkpoint_path, latest_file)

        if os.path.exists(checkpoint_file_path):
            log(INFO, f"Loading checkpoint from {checkpoint_file_path}")
            weights = np.load(checkpoint_file_path)
            weights = [val for _, val in weights.items()]
            model = set_weights(model, weights)

    return model


def get_weights(model: YOLO) -> NDArrays:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model
