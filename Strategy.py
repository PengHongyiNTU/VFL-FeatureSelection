from abc import ABC, abstractmethod
from pickle import NONE
from Client import Client
from typing import Sequence
import torch

from Courier import Courier

class Strategy(ABC):
    def __init__(self, courier:Courier, clients:Sequence[Client]):
        self.courier = courier
        self.clients = clients
    @abstractmethod
    def aggregate(self):
        pass
    @abstractmethod
    def update_all(self):
        pass

class SyncConcatStrategy(Strategy):
    def __init__(self, courier, clients):
        super().__init__(courier, clients)

    def is_all_set(self):
        return not any(elem is None for elem in self.courier.message_pool.values())
    
    def update_all(self, loss):
        loss.backward()
        map(lambda client: client.update(), self.clients)

    def aggregate(self, eval=False):
        # Check all clients sent their embedding
        if not eval:
            for client in self.clients:
                client.fit()
        else:
            for client in self.clients:
                client.predict()

        # print(self.courier.message_pool.values())         
        if self.is_all_set():
            emb_list = list(self.courier.message_pool.values())
            embs = torch.cat(emb_list, 1)
            self.courier.flush()
            return embs
    



class SyncSTGConcatStrategy(SyncConcatStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_all(self, loss, server_reg_loss):
        reg_loss = server_reg_loss
        for client in self.clients:
            reg_loss += client.model.get_reg_loss()
        reg_loss = reg_loss/(len(self.clients)+1)
        total_loss = reg_loss + loss
        total_loss.backward()
        map(lambda client: client.update(), self.clients)
    
    def number_of_features(self):
        num_feats = 0
        for client in self.clients:
            _, num = client.model.get_gates()
            num_feats += num
        return num_feats