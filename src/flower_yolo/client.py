import os
import sys
from typing import OrderedDict
import flwr as fl

from logging import INFO
from flwr.client import NumPyClient
from flwr.common.logger import log
from flwr.common import NDArrays

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from utils import get_weights, load_model


class RobotClient(NumPyClient):
    def __init__(self, id, data, epochs):
        self.model: YOLO = load_model()
        self.client_id = id
        self.data = data
        self.epochs = epochs

    def fit(self, parameters: NDArrays, config):
        self.set_weights(parameters)
        results: DetMetrics | None = self.model.train(
            data=self.data,
            epochs=self.epochs,
            name=self.client_id,
        )
        if results is not None:
            log(INFO, f"Client {self.client_id} trained model")
            log(INFO, f"Metrics: {results}")

        return get_weights(self.model), 4, {}

    def evaluate(self, parameters: NDArrays, config):
        self.set_weights(parameters)
        matrics: DetMetrics = self.model.val()
        accuracy = matrics.box.map
        loss = matrics.fitness

        return loss, 4, {"accuracy": accuracy}

    def set_weights(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)


def client_fn(id):
    data = "coco8.yaml"
    epochs = 1

    return RobotClient(id, data, epochs).to_client()


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        raise ValueError("Expected exactly one argument: client id")

    client_id = args[1]

    log(INFO, f"Client ID: {client_id}")

    server_ip = os.getenv("SERVER_IP", "localhost")
    server_port = os.getenv("SERVER_PORT", "8080")

    log(INFO, f"Server IP: {server_ip}:{server_port}")

    fl.client.start_client(
        server_address=f"{server_ip}:{server_port}",
        client=client_fn(client_id),
    )
