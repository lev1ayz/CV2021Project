"""Define your architecture here."""
from turtle import forward
import torch
from torch import nn
from models import SimpleNet, generate_xception_head_mlp
import torchvision.models as torch_models
from torchvision.models.resnet import model_urls
from collections import OrderedDict

def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = Resnet30BasedModel()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model

class Resnet30BasedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch_models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*(list(self.encoder.children())[:-1]))

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        mlp = [
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        ] 

        self.fc = nn.Sequential(*mlp)

    def forward(self, x):
        h = self.encoder(x).squeeze()
        out = self.fc(h)
        return out