# Clients
from Client import SyncFNNClient
from torch import nn
import numpy as np
import Courier
from VFLDataUtils import SimpleNumpyDataLoader
from Strategy import SyncConcatStrategy
from Server import SyncFNNServer



clients_id_list = list(range(3))
x, y = np.random.randn(2000, 400), np.ones(2000)
courier = Courier.SyncLocalCourier(clients_id_list)
loader = SimpleNumpyDataLoader(
    clients_id_list=clients_id_list,
    data_source=(x, y), dim_split=[100, 200, 300]
)
loader_dict = loader.distribute()
print(loader_dict)


clients = [
    SyncFNNClient(id=id, 
    model=nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU()),
    courier=courier,
    train_loader=loader_dict[id]['train_loader'],
    test_loader=loader_dict[id]['test_loader'],
    config_dir='simple_config.ini')
for id in clients_id_list]

strategy = SyncConcatStrategy(courier=courier, clients=clients)
server =  SyncFNNServer(
    strategy=strategy,
    courier=courier,
    top_model=nn.Sequential(
        nn.Linear(4 * 32, 32),
        nn.ReLU(True),
        nn.Linear(32, 1),
        nn.Sigmoid()),
    emb_model=nn.Sequential( 
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU()),
    train_loader=loader_dict['server']['train_loader'],
    test_loader=loader_dict['server']['test_loader'],
    config_dir='simple_config.ini')

####### Start Training ############
print('Training Starts')
print('-'*89)
for epoch in range(5):
    server.fit(epoch)
print(server.get_history())

