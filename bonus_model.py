"""Define your architecture here."""
import torch
from torch import nn
import torchvision.models as torch_models

def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = ResnetBasedModel()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model

class ResnetBasedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch_models.resnet18(pretrained=True)
        print(self.encoder)
        self.encoder = nn.Sequential(*(list(self.encoder.children())[:-1]))

        mlp = [
            nn.Linear(512, 2),
        ] 

        self.fc = nn.Sequential(*mlp)

    def forward(self, x):
        h = self.encoder(x).squeeze()
        out = self.fc(h)
        return out