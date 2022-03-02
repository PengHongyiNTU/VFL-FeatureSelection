from http import server
from torch.utils.data import DataLoader
import torch
from torch import nn
from Courier import Courier, SyncLocalCourier
from Strategy import Strategy, SyncConcatStrategey
from abc import ABC, abstractmethod
import configparser
import pandas as pd

class Server(ABC):
    def __init__(self, strategy:Strategy, courier:Courier, model:nn.Module,
        train_loader:DataLoader, test_loader:DataLoader, config_dir:str 
    ):
        self.strategy = strategy
        self.courier = courier
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = configparser.ConfigParser.read(config_dir)['Server']

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        pass



class SyncFNNServer(server):
    def __init__(self, strategy:SyncConcatStrategey, courier:SyncLocalCourier, 
    model, train_loader, test_loader, config_dir):
        super().__init__(strategy, courier, model, train_loader, 
        test_loader, config_dir)
        self.train_acc = []
        self.train_loss = []
        self.test_acc = []
        if self.config['Optimizer'] == 'adam':
            lr = float(self.config['LearningRate'])
            self.optimizer = torch.optim.Adam(self.model.parameters())
        if self.config['Device'] == 'cude':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')   
        
    
    def fit(self, num_epochs):
        for e in range(num_epochs):
            train_acc = 0
            train_loss = 0
            

    def evaluate(self, num_epochs):
        pass

    def get_history(self):
        return self.history 


