from typing import  Sequence, TypedDict, Mapping
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import trian_test_spilt
from VFLDataset import VFLDataLoader, LoaderDict
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

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
    train_batch_size=128, test_batch_size=1000):
        super().__init__(clients_id_list, data_source)
        self.dict = dict.fromkeys(self.clients_list, None)
        self.x, self.y = data_source[0], data_source[1]
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def __preprocess__(self):
        x_train, x_test, y_train, y_test = trian_test_spilt(
            self.x, self.y, test_size=0.2, random_state=42)
        feat_dim = self.x.shape[0]
        num_clients = len(self.clients_list)
        feat_idx_list =  np.array_split(np.arange(feat_dim), 
        num_clients)
        for i, clients_id in enumerate(self.dict.keys()):
            feat_idx = feat_idx_list[i]
            self.dict[clients_id] = {
                'train_loader': 
                 DataLoader(TensorDataset(
                    (self.x_train[:, feat_idx], self.y_train), 
                    batch_size = self.train_batch_size
                )), 
                'test_loader':
                DataLoader(
                    TensorDataset
                )

            }


    def distribute(self):
        pass
