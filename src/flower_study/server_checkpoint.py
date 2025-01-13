import os
from typing import List, Tuple, Union, Optional, Dict

import flwr as fl
import numpy as np
from logging import INFO
from flwr.common.logger import log
from numpy.typing import NDArray
from flwr.common import (
    Parameters,
    Scalar,
    Metrics,
    ndarrays_to_parameters,
    FitRes,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from utils import load_model_with_checkpoint


class SaveModelStrategy(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[NDArray] = parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays
            log(INFO, f"Saving round {server_round} aggregated_ndarrays...")

            checkpoint_path: str = os.path.join(os.getcwd(), "checkpoints")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            if os.getenv("KUBERNETES_SERVICE_HOST") is not None:
                checkpoint_path = "/mnt/checkpoints/"

            np.savez(
                f"{checkpoint_path}/round-{server_round}-weights.npz",
                *aggregated_ndarrays,
            )

        return aggregated_parameters, aggregated_metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}  # type: ignore


def server_fn():
    """Construct components that set the ServerApp behaviour."""

    # Let's define the global model and pass it to the strategy
    # Note this is optional.
    parameters = ndarrays_to_parameters(
        load_model_with_checkpoint().get_weights(),
    )

    # Define the strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.5,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    # Read from config
    num_rounds = 3
    config = ServerConfig(num_rounds=num_rounds)

    return strategy, config


if __name__ == "__main__":
    strategy, config = server_fn()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=config,
    )
