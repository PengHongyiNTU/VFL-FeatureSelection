from abc import ABC, abstractmethod
from typing import Union
import torch
import numpy as np

class Courier(ABC):
    def __init__(self, clients_list) -> None:
        self.clients_list = clients_list
        self.message_pool = dict.fromkeys(clients_list, None)
        self.response_pool = dict.fromkeys(clients_list, None)
    
    @abstractmethod
    def respond(self, id, gradient:torch.tensor)->None:
        pass
    @abstractmethod
    def post(self, id:Union[str, int], message) -> None:
        pass
    @abstractmethod
    def fetch(self, id:Union[str, int]) -> Union[torch.tensor, str, np.array]:
        pass
    @abstractmethod
    def flush(self) -> None:
        pass



# Synchornized
class SyncLocalCourier(Courier):
    def __init__(self, clients_list):
        super().__init__(clients_list)

    def communication_overhead(self):
        # Compute the size of each tensor in the mesage pool
        pass


    def respond(self, id, gradient:torch.tensor)->None:
        self.response_pool[id] = gradient

    def post(self, id, message:torch.tensor):
        self.message_pool[id] = message
    
    def fetch(self, id):
        res = self.response_pool[id]
        return res
    
    def flush(self):
        map(lambda x: x.clear(), self.response_pool.values())
        map(lambda x: x.clear(), self.message_pool.values())


class InSyncLocalCourier(Courier):
    def __init__(self, clients_list):
        super().__init__(clients_list)

    def post(self, id, message):
        super().post(id, message)

    def fetch(self, id):
        super().fetch(id)

    def flush(self):
        super().flush()


class InSyncNetworkSocketCourier(Courier):
    def __init__(self, clients_list):
        super().__init__(clients_list)

    def post(self, id, message):
        super().post(id, message)

    def fetch(self, id):
        super().fetch(id)

    def flush(self):
        super().flush()



