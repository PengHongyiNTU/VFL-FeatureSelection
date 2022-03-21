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
        self.config = configparser.ConfigParser()
        self.config.read(config_dir)
        self.config = self.config['Server']

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        pass



class SyncFNNServer(Server):
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
        if self.config['Device'] == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')   
        self.model = self.model.to(self.device)
        self.emb_model = self.emb_model.to(self.device)


    def binary_acc(self, out, y):
        acc = accuracy_score(y, out>0.5)
        return acc
        
    
    def fit(self, e):
        train_acc = 0
        train_loss = 0
        for x, y in self.train_loader:
            x = x.float().to(self.device)
            y = y.float().to(self.device).view(-1, 1)
            self.optimizer.zero_grad()
            server_emb = self.emb_model(x)
            clients_emb = self.strategy.aggregate()
            # print('server-emb', server_emb)
            # print('client-emb', clients_emb)
            # print('embshape',server_emb.shape)
            # print('clients-embshape', clients_emb.shape)
            # Notify all clients to optimize their model
            emb = torch.cat([server_emb, clients_emb], 1)
            out = self.model(emb)
            loss = self.criterion(out, y)
            acc = self.binary_acc(out.detach().cpu(), y.detach().cpu())
        
            # In Sync setting, Gradient will automatically propogate back
            # Can also compute the gradient from message_pool again
            # Notify all clients 
            # loss.backward()
            self.strategy.update_all(loss)
            self.optimizer.step()
            train_loss += loss.item()
            train_acc += acc

        test_acc = self.evaluate()
        print(f'Epoch {e+0:03}: | Loss: {train_loss/len(self.train_loader):.5f} | Acc: {train_acc/len(self.train_loader):.3f} | Val ACC: {test_acc/len(self.test_loader):.3f}')
        self.train_acc.append(train_acc/len(self.train_loader))
        self.test_acc.append(test_acc/len(self.test_loader))

            

    def evaluate(self):
        self.emb_model.eval()
        self.model.eval()
        test_acc = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                server_emb = self.emb_model(x)

                clients_emb = self.strategy.aggregate(eval=True)
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








class SyncSTGServer(SyncFNNServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    


    def fit(self, e):
        train_acc = 0
        train_loss = 0
        for x, y in self.train_loader:
            x = x.float().to(self.device)
            y = y.float().to(self.device).view(-1, 1)
            self.optimizer.zero_grad()
            server_emb = self.emb_model(x)
            clients_emb = self.strategy.aggregate()
            emb = torch.cat([server_emb, clients_emb], 1)
            server_reg_loss = self.emb_model.get_reg_loss()
            out = self.model(emb)
            loss = self.criterion(out, y)
            acc = self.binary_acc(out.detach().cpu(), y.detach().cpu())
            self.strategy.update_all(loss, server_reg_loss)
            self.optimizer.step()
            train_loss += loss.item()
            train_acc += acc
        test_acc = self.evaluate()
        _, num_feats = self.emb_model.get_gates()
        num_feats += self.strategy.number_of_features()
        

        print(f'Epoch {e+0:03}: | Loss: {train_loss/len(self.train_loader):.5f} | Acc: {train_acc/len(self.train_loader):.3f} | Val ACC: {test_acc/len(self.test_loader):.3f} | Features Left: {num_feats}')
        self.train_acc.append(train_acc/len(self.train_loader))
        self.test_acc.append(test_acc/len(self.test_loader))
