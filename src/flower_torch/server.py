from typing import List, Tuple, Union, Optional, Dict, OrderedDict

import flwr as fl
from logging import INFO
from flwr.common.logger import log
from flwr.common import (
    Parameters,
    Scalar,
    Metrics,
    ndarrays_to_parameters,
    FitRes,
)
from flwr.server import ServerConfig, client_proxy
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
import numpy as np
import torch

from utils import Net, get_weights


class SaveModelStrategy(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            net = Net()
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}  # type: ignore


def server_fn():
    model = Net()
    parameters = ndarrays_to_parameters(get_weights(model))

    log(INFO, "Initial parameters")

    strategy = SaveModelStrategy(
        fraction_fit=0.5,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

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
