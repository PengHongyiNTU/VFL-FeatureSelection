from http import server
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch
from torch import nn
from Courier import Courier, SyncLocalCourier
from Strategy import Strategy, SyncConcatStrategy
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
    def __init__(self, strategy:SyncConcatStrategy, courier:SyncLocalCourier, 
    top_model, emb_model, train_loader, test_loader, config_dir,
    criterion=nn.BCELoss()):
        super().__init__(strategy, courier, top_model, train_loader, 
        test_loader, config_dir)
        self.emb_model = emb_model
        self.criterion = criterion
        self.train_acc = []
        self.test_acc = []
        if self.config['Optimizer'] == 'adam':
            lr = float(self.config['LearningRate'])
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()) + 
                list(self.emb_model.parameters()), 
                lr = lr)
        if self.config['Device'] == 'cude':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')   
        self.model = self.model.to(self.device)
        self.emb_model = self.emb_model.to(self.device)


    def binary_acc(out, y):
        acc = accuracy_score(y, out>0.5)
        return acc
        
    
    def fit(self, e):
        train_acc = 0
        train_loss = 0
        for x, y in self.train_loader:
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            self.optimizer.zero_grad()
            server_emb = self.emb_model(x)
            clients_emb = self.strategy.aggreate()
            # Notify all clients to optimize their model
            emb = torch.cat([server_emb, clients_emb], 1)
            out = self.model(emb)
            loss = self.criterion(out, y)
            acc = self.binary_acc(out.detach().cpu(), y.detach().cpu())
        
            # In Sync setting, Gradient will automatically propogate back
            # Can also compute the gradient from message_pool again
            # Notify all clients 
            loss.backward()
            self.courier.server_done = True
            self.optimizer.step()
            train_loss += loss.item()
            train_acc += acc
        test_acc = self.evaluate()
        print(f'Epoch {e+0:03}: | Loss: {train_loss/len(self.train_loader):.5f} | Acc: {train_acc/len(self.train_loader):.3f} | Val ACC: {test_acc/len(self.test_loader):.3f}')
        self.train_acc.append(train_acc/len(self.train_loader))
        self.test_acc.append(test_acc/len(self.test_loader))

            

    def evaluate(self, num_epochs):
        self.emb_model.eval()
        self.model.eval()
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                server_emb = self.emb_model(x)
                clients_emb = self.strategy.aggreate()
                emb = torch.cat([server_emb, clients_emb], 1)
                out = self.model(emb)
                acc = self.binary_acc(out.detach().cpu(), y.detach().cpu())
                test_acc += acc
        return test_acc

    def get_history(self):
        self.history = pd.DataFrame({
            'train-acc': self.train_acc,
            'test-acc': self.test_acc
        })
        return self.history 


