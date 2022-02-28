from pyexpat import model
from unittest import TestLoader
import flwr as fl
from torch.utils.data import DataLoader
import torch
from torch import nn
from zmq import device
class TorchFNNClient(fl.client.NumPyClient):
    # cid: integer that identify client
    # model: client model should be a instance of nn.Module
    # train_loader: Training Data Loader
    # test_loader: Test Data Loader
    # epochs: int
    # device: cpu or cuda
    def __init__(self, cid:int, model:nn.Module, 
    train_loader:DataLoader, test_loader:TestLoader,
    epochs:int, device:torch.device("cpu")) -> None:
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
    
    # Passing Weights as message to the Server
    # Weights here should be the embedding
    def get_weights(self, x) -> fl.common.Weights:
        embedding = self.model(x)
        return [embedding.cpu()]

    # Receive parameters 
    def set_parameters(self, parameters):
        print(parameters)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, epochs=1)
        return self.get_parameters(), 


        
