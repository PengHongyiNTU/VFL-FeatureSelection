from typing import  Sequence, TypedDict, Mapping
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch

class LoaderDict(TypedDict):
    train_loader: DataLoader
    test_loader: DataLoader


class VFLDataLoader(ABC):
    def __init__(self, clients_list:Sequence, data_source):
        self.clients_list = clients_list
        self.data_source = data_source
    
    # Handle all the Preprocessing here 
    # Including load the dataset, splict the dataset
    @abstractmethod
    def __preprocess__(self) -> None:
        pass

    # Distribute dataloader to each client in clients_list
    # Return a dictionary {clients_id: (train)}    
    @abstractmethod
    def distribute(self) -> Mapping[int, LoaderDict]:
        pass





class SimpleNumpyDataLoader(VFLDataLoader):
    def __init__(self, clients_id_list,  data_source:Sequence[np.array], 
    train_batch_size=128, test_batch_size=1000, dim_split=None):
        super().__init__(clients_id_list, data_source)
        self.dict = dict.fromkeys(self.clients_list, None)
        self.x, self.y = data_source[0], data_source[1]
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.dim_split = dim_split

    def __preprocess__(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42)
        feat_dim = self.x.shape[1]
        num_clients = len(self.clients_list)
        if not self.dim_split:
            feat_idx_list =  np.array_split(np.arange(feat_dim),   num_clients+1)
            for i, feat_idx in enumerate(feat_idx_list[:-1]):
                print(f'Client {i}: Feature Index {feat_idx[0]}-{feat_idx[-1]}')
            server_idx = feat_idx_list[-1]
            print(f'Server : Feature Index {server_idx[0]}-{server_idx[-1]}')


        else:
            feat_idx_list = []
            start = 0
            assert len(self.dim_split) == num_clients and self.dim_split[-1] < feat_dim
            for i, split in enumerate(self.dim_split):
                feat_idx_list.append(
                    np.arange(feat_dim)[start: split]
                )
                print(f'Client {i}: Feature Index {start}-{split}')
                start = split

            feat_idx_list.append(np.arange(feat_dim)[start:])
            print(f'Server : Feature Index {start}-{feat_dim}')
        assert len(feat_idx_list) == num_clients+1
        for i, clients_id in enumerate(self.dict.keys()):
            feat_idx = feat_idx_list[i]
            self.dict[clients_id] = {
                'train_loader': 
                 DataLoader(TensorDataset(
                    torch.tensor(x_train[:, feat_idx])), 
                    batch_size = self.train_batch_size,
                    shuffle=False
                ), 
                'test_loader':
                DataLoader(
                    TensorDataset(
                        torch.tensor(x_test[:, feat_idx])), 
                        batch_size = self.test_batch_size,
                        shuffle=False
                    ) 

            }
        server_idx = feat_idx_list[-1]
        self.dict['server'] = {
            'train_loader':  DataLoader(TensorDataset(
                    torch.tensor(x_train[:, server_idx]), 
                    torch.tensor(y_train)), 
                    batch_size = self.train_batch_size,
                    shuffle=False
                ), 
            'test_loader':
                DataLoader(
                    TensorDataset(
                        torch.tensor(x_test[:, feat_idx]), 
                        torch.tensor(y_test)), 
                        batch_size = self.test_batch_size,
                        shuffle=False
                    ) 
        }

    def distribute(self):
        self.__preprocess__()
        return self.dict