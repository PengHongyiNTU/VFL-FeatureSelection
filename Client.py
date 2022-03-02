
from torch.utils.data import DataLoader
import torch
from torch import nn
from typing import Union
from Courier import Courier
from abc import ABC, abstractmethod
import configparser

class Client(ABC):
    def __init__(self, id:Union[int, str], 
    model:nn.Module, 
    courier:Courier, train_loader:DataLoader, 
    test_loader:DataLoader, config_dir:str) -> None:
        self.id = id
        self.model = model
        self.courier = courier
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = configparser.ConfigParser().read(config_dir)

    @abstractmethod
    def fit(self):
        pass
    @abstractmethod
    def evaluate(self):
        pass
    @abstractmethod
    def send(self, message):
        pass
    @abstractmethod
    def recv(self):
        pass
    @abstractmethod
    def is_available(self):
        pass

        



class SyncFNNClient(Client):
    def __init__(self, id, model, courier, train_loader, 
    test_loader, config_dir):
        super().__init__(id, model, courier, train_loader, 
    test_loader, config_dir)
        self.client_config = self.config['Client']
        if self.client_config['Optimizer'] == 'adam':
            lr = float(self.client_config['LearningRate'])
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.client_config['Device'] == 'cuda':
            self.device = torch.device('cuda') 
        else:
            self.device = torch.device('cpu')

    def is_available(self):
        return True

    def send(self, message):
        if self.is_available():
            self.courier.post(self.id, message)
    
    def recv(self):
        if self.is_available():
            return self.courier.fetch(self.id)

    def fit(self):
        if self.is_available():
            self.model.train()
            self.model.to(self.device)
            for x in self.train_loader:
                x = x.float().to(self.device)
                self.optimizer.zero_grad()
                emb = self.model(x)
                self.send(emb)
                # Simple Concat Dont Require Fetched Gradient
                res = self.recv()
                # Speed up the gradient calculation
                # Don't recompute the gradient since the SimpleConcatStrategy is used
                self.optimizer.step()



            

        




        
