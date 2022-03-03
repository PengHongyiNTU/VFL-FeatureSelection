from abc import ABC, abstractmethod
from Client import Client
from typing import Sequence
import torch

from Courier import Courier

class Strategy(ABC):
    def __init__(self, courier:Courier, clients:Sequence[Client]):
        self.courier = courier
    @abstractmethod
    def aggregate(self):
        pass

class SyncConcatStrategy(Strategy):
    def __init__(self, courier, clients):
        super().__init__(courier, clients)

    def is_all_set(self):
        bool([message for message in self.courier.message_pool.values() if message != []])
   
    def aggregate(self, eval=False):
        # Check all clients sent their embedding
        if not eval:
            for client in self.clients:
                client.fit()
        if eval:
            for client in self.clients:
                client.predict()
        if self.is_all_set():
            emb_list = self.courier.message_pool.values()
            embs = torch.cat(emb_list, 1)
            self.courier.flush()
            return embs
    
