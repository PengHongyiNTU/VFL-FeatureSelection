from typing import Mapping, Sequence, Tuple
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


class VFLDataLoader(ABC):
    def __init__(self, clients_list:Sequence, data_source:Dataset):
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
    def distribute(self) -> Mapping[int, Tuple[DataLoader, ...]]:
        pass

