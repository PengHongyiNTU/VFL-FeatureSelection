from abc import ABC, abstractmethod
import torch

from Courier import Courier

class Strategy(ABC):
    def __init__(self, courier:Courier):
        self.courier = courier
    @abstractmethod
    def aggregate(self):
        pass

class SyncConcatStrategey(Strategy):
    def __init__(self, courier):
        super().__init__(courier)

    def is_all_set(self):
        bool([message for message in self.courier.message_pool.values() if message != []])
   
    def aggregate(self):
        # Check all clients sent their embedding
        if self.is_all_set():
            emb_list = self.courier.message_pool.values()
            embs = torch.cat(emb_list, 1)
            return embs
    
