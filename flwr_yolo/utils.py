import os
from typing import Dict, Optional
import yaml
from glob import glob
import numpy as np
from logging import DEBUG, ERROR, INFO, WARNING
from flwr.common.logger import log
from collections import OrderedDict

import torch

from ultralytics.models import YOLO

from flwr.common import NDArrays


def load_model():
    # model = YOLO("yolo11x.pt")
    model = YOLO("former_best.pt")
    return model


def load_model_with_checkpoint() -> YOLO:
    model = load_model()

    checkpoint_path: str = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if os.getenv("KUBERNETES_SERVICE_HOST") is not None:
        checkpoint_path = "/mnt/checkpoints/"

    list_of_files = [fname for fname in glob(checkpoint_path + "/*.npz")]

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


def set_weights(model: YOLO, parameters: NDArrays):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict=state_dict, strict=True)
    return model


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_batch() -> int:
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / (1024**3)  # Convert to GB

        log(INFO, f"GPU memory ({total_memory_gb:.2f} GB) is available.")

        if total_memory_gb < 6:
            log(
                INFO,
                f"GPU memory ({total_memory_gb:.2f} GB) is less than 6GB. Setting batch size to 4.",
            )
            return 4
        else:
            return -1  # Use default batch size from YOLO
    else:
        return 4  # Default to a smaller batch size for CPU or if GPU memory check fails


def get_data_image_count(yaml_path: str) -> Optional[int]:
    try:
        with open(yaml_path, "r") as file:
            data: Dict[str, str] = yaml.safe_load(file)
            dataset_path = os.path.join(
                os.getcwd(), data.get("path", "").replace("./", "")
            )
            log(INFO, f"Dataset path: {dataset_path}")
            if dataset_path:
                train_path = os.path.join(dataset_path, data.get("train", ""))
                log(DEBUG, f"Train path: {train_path}")
                if os.path.exists(train_path):
                    image_files = glob(os.path.join(train_path, "*.*"))
                    log(DEBUG, f"Image count: {len(image_files)}")
                    return len(image_files)

    except FileNotFoundError:
        log(ERROR, f"File not found: {yaml_path}")
    except Exception as e:
        log(ERROR, f"An error occurred: {e}")

    log(WARNING, "No image files found.")
    return None
