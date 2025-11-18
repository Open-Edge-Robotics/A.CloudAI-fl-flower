import os
import fire
import yaml
from glob import glob
from logging import DEBUG, ERROR, INFO, WARNING
from flwr.common.logger import log

from typing import Dict, Optional


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


if __name__ == "__main__":
    fire.Fire({"count": get_data_image_count})
