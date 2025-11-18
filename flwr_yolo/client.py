from logging import ERROR, INFO
from flwr.client import NumPyClient, ClientApp
from flwr.common import log, Context, NDArrays, Scalar

from ultralytics.models import YOLO
from ultralytics.utils.metrics import DetMetrics

from flwr_yolo.utils import (
    get_weights,
    load_model,
    set_weights,
    get_device,
    get_batch,
    get_data_image_count,
)

from typing import Dict, Tuple


class RobotClient(NumPyClient):

    def __init__(self, id: bool | float | int | str, data: str, num_examples: int):
        self.model: YOLO = load_model()
        self.client_id = id
        self.data = data
        self.num_examples = num_examples
        self.device = get_device()
        self.batch = get_batch()

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        log(INFO, f"Client {self.client_id} training model")
        log(INFO, f"Config: {config}")
        try:
            self.model = set_weights(self.model, parameters)
            results: DetMetrics | None = self.model.train(
                project="flower-yolo-train",
                name=f"Client_{self.client_id}",
                data=self.data,
                epochs=config["epochs"],
                batch=self.batch,
                workers=0,
                device=self.device,
            )

            if results is not None:
                log(INFO, f"Client {self.client_id} trained model")
                return (
                    get_weights(self.model),
                    self.num_examples,
                    {
                        "accuracy": results.box.map,
                        "loss": results.fitness,
                    },
                )

        except Exception as e:
            log(ERROR, f"Client {self.client_id} failed to train model: {e}")

        return (
            get_weights(self.model),
            self.num_examples,
            {"accuracy": 0.0, "loss": 0.0},
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, dict[str, Scalar]]:
        try:
            self.model = set_weights(self.model, parameters)
            matrics: DetMetrics = self.model.val(
                project="flower-yolo-val",
                name=f"Client_{self.client_id}",
                data=self.data,
                workers=0,
                device=self.device,
            )
            if isinstance(matrics, DetMetrics):
                log(INFO, f"Client {self.client_id} evaluated model")
                accuracy: float = matrics.box.map
                loss: float = matrics.fitness
                return loss, self.num_examples, {"accuracy": accuracy}

        except Exception as e:
            log(ERROR, f"Client {self.client_id} failed to evaluate model: {e}")

        return 0.0, self.num_examples, {"accuracy": 0.0}


def client_fn(context: Context):
    client_id = context.node_config["client_id"]
    data = f"former_{client_id}.yaml"
    num_examples = get_data_image_count(data)

    log(INFO, "=" * 50)
    log(INFO, f"Client ID: {client_id}")
    log(INFO, "=" * 50)

    assert num_examples is not None
    return RobotClient(id=client_id, data=data, num_examples=num_examples).to_client()


app = ClientApp(client_fn=client_fn)
