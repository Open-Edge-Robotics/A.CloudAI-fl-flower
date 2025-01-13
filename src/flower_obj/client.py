from logging import INFO
import os
import sys
import torch

import flwr as fl
from flwr.common.logger import log
from flwr.client import NumPyClient
from flwr.common import Context, RecordSet

from model import RCNNModel, get_weights, load_data, set_weights, eval, train


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        model_instance = RCNNModel()
        self.model = model_instance.get_model_instance_segmentation(2)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.loss = 0.0

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.model, parameters)
        self.loss = train(
            self.model,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.model), len(self.trainloader.dataset), {"loss": self.loss}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.model, parameters)
        loss, accuracy = eval(self.model, self.valloader, self.device)
        return self.loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]

    trainloader, valloader = load_data(batch_size=int(batch_size))

    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()


if __name__ == "__main__":
    args = sys.argv[1:]
    node_id = int(args[0]) if args else 1

    server_ip = os.getenv("SERVER_IP", "localhost")
    server_port = os.getenv("SERVER_PORT", "8080")

    log(INFO, f"Server IP: {server_ip}:{server_port}")

    fl.client.start_client(
        server_address=f"{server_ip}:{server_port}",
        client=client_fn(
            context=Context(
                node_id=node_id,
                state=RecordSet(),
                node_config={"partition-id": 0, "num-partitions": 1},
                run_config={
                    "num_classes": 2,
                    "batch-size": 1,
                    "local-epochs": 1,
                    "learning-rate": 0.01,
                },
            ),
        ),
    )
