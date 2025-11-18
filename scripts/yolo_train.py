import fire

from utils import load_model_with_checkpoint


class YoloTrain:
    def __init__(self, model_name: str, dataset: str, batch: int) -> None:
        self.model = load_model_with_checkpoint(model_name)
        self.dataset = dataset
        self.batch = batch

    def train(self):
        self.model.train(data=self.dataset, epochs=10, imgsz=320, batch=self.batch)


def train(
    model_name: str = "yolo11x.pt",
    dataset: str = "coco128_custom.yaml",
    batch: int = 4,
):
    yolo_train = YoloTrain(model_name, dataset, batch)
    yolo_train.train()


if __name__ == "__main__":
    fire.Fire({"train": train})
