
from models.model import AnomalyModel
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode
    
    
class AutoencoderModel(AnomalyModel):
    """ Class for Autoencoder model"""
    def __init__(self, config):
        super().__init__(config)
    