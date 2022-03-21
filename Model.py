from stg.models import MLPLayer, FeatureSelector
from torch import nn
import torch 


class STGEmbModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, sigma=1.0, lam=0.1):
        super().__init__()
        self.mlp = MLPLayer(input_dim, output_dim, hidden_dims, 
        batch_norm=None, dropout=None, activation='relu', flatten=True)
        self.fs = FeatureSelector(input_dim, sigma, device=torch.device('cuda'))
        self.reg = self.fs.regularizer
        self.lam = lam
        self.mu = self.fs.mu
        self.sigma = self.fs.sigma
    
    def forward(self, x):
        x = self.fs(x)
        emb = self.mlp(x)
        return emb
    
    
    def count_num_features(self)-> int:
        with torch.no_grad():
            for name, param in self.fs.named_parameters():
                if name == "mu":
                    return param.nonzero().size(0)
                
                    

    def get_mu(self):
        return self.mu.deatach().cpu().numpy()
    


def make_models(input_dims):
    models = []
    for input_dim in input_dims:
        model = []
        if input_dim >= 512:
            model.append(nn.Linear(input_dim, 512))
            model.append(nn.ReLU())
            model = model + [nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()]
        else:
            model = [nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()]
        models.append(nn.Sequential(*model))
    top_model = nn.Sequential(
        nn.Linear(len(input_dims)*128, 32), nn.ReLU(True), nn.Linear(32, 1), nn.Sigmoid()
    )
    return models, top_model




def make_stg_models(input_dims):
    models = []
    for input_dim in input_dims:
        model = []
        if input_dim >= 512:
            model = STGEmbModel(input_dim=input_dim, output_dim=128, hidden_dims=[512, 256])
        else:
            model = STGEmbModel(input_dim=input_dim, output_dim=128, hidden_dims=[256])
        models.append(model)
    top_model = nn.Sequential(
        nn.Linear(len(input_dims)*128, 32), nn.ReLU(True), nn.Linear(32, 1), nn.Sigmoid()
    )
    return models, top_model



if __name__ == '__main__':
    model = STGEmbModel(input_dim=512, output_dim=128, hidden_dims=[512, 256])
    x = torch.ones(100, 512)
    y = model(x)
    # print(y.shape)s