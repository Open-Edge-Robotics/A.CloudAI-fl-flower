import numpy as np
import os

from logging import INFO
from flwr.common.logger import log
from flwr.common import (
    Parameters,
    Scalar,
    Metrics,
    ndarrays_to_parameters,
    FitRes,
    Context,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig, ServerAppComponents, ServerApp
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common.typing import NDArrays

from flwr_yolo.utils import get_weights, load_model_with_checkpoint

from typing import List, Tuple, Union, Optional, Dict


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
            aggregated_ndarrays: NDArrays = parameters_to_ndarrays(
                aggregated_parameters
            )

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


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    log(INFO, f"Metrics: {metrics}")
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}  # type: ignore


def server_fn(context: Context):
    log(INFO, f"Server context: {context}")
    model = load_model_with_checkpoint()
    parameters = ndarrays_to_parameters(get_weights(model))

    min_available_clients = context.run_config["min-available-clients"]
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    epochs = context.run_config["epochs"]

    strategy = SaveModelStrategy(
        initial_parameters=parameters,
        fraction_fit=float(fraction_fit),
        fraction_evaluate=float(fraction_evaluate),
        min_available_clients=int(min_available_clients),
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda _: {
            "learning-rate": 0.1,
            "epochs": epochs,
            "batch_size": 16,
        },
    )

    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
