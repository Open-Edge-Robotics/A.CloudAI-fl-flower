import os
import sys
import flwr as fl
from logging import INFO
from flwr.client import NumPyClient
from flwr.common.logger import log
from utils import load_custom_dataset, load_data, load_model_with_checkpoint  # noqa


# Define Flower Client
class RobotClient(NumPyClient):
    def __init__(
        self,
        data,
        epochs,
        batch_size,
        verbose,
    ):
        self.model = load_model_with_checkpoint()
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def get_parameters(self, config):
        """Return the parameters of the model of this client."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose="auto",
        )
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(client_id: str):
    """Construct a Client that will be run in a ClientApp."""

    data = load_custom_dataset(client_id)

    epochs = 10
    batch_size = 32
    verbose = False

    # Return Client instance
    return RobotClient(
        data,
        epochs,
        batch_size,
        verbose,
    ).to_client()


if __name__ == "__main__":
    server_ip = os.getenv("SERVER_IP", "localhost")
    server_port = os.getenv("SERVER_PORT", "8080")

    args = sys.argv
    if len(args) > 1:
        client_id = args[1]

    log(INFO, f"Server IP: {server_ip}:{server_port}")

    fl.client.start_client(
        server_address=f"{server_ip}:{server_port}",
        client=client_fn(client_id),
    )
