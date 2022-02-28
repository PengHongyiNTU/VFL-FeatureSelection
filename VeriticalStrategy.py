import flwr as fl
from flwr.server.strategy import Strategy
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class VerticalStrategy(Strategy):
    def __init__(self):
        pass
    
    def configure_fit(self, rnd: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        return super().configure_fit(rnd, parameters, client_manager)

    def configure_evaluate(self, rnd: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return super().configure_evaluate(rnd, parameters, client_manager)

    def aggregate_evaluate(self, rnd: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[BaseException]) -> Union[Tuple[Optional[float], Dict[str, Scalar]], Optional[float]],:
        return super().aggregate_evaluate(rnd, results, failures)
    
    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return super().evaluate(parameters)
