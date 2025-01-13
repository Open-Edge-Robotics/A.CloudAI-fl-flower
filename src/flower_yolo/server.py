from typing import List, Tuple, Union, Optional, Dict

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
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from utils import get_weights, load_model


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

        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    log(INFO, f"Metrics: {metrics}")
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}  # type: ignore


def server_fn():
    model = load_model()
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
