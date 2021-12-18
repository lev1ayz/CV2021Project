"""Define your architecture here."""
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
    model = get_simclr_based_model()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('./checkpoints/bonus_model.pt')['model'])
    return model
'''
def get_simclr_based_model() -> nn.Module:
    simclr = ResNet50()
    #state_dict = torch.utils.model_zoo.load_url(model_urls['ResNet50'.lower()])
    ckpt = torch.load('./checkpoints/resnet50_imagenet_bs2k_epochs200.pth.tar')
    print(f'ckpt keys:{ckpt.keys()}\nhparams:{ckpt["hparams"]}')
    #ckpt['hparams'].encoder_ckpt
    simclr.load_state_dict(ckpt['state_dict'])

    simclr.fc = generate_xception_head_mlp()
    print(f'simclr:\n{simclr}')
    # TODO set base to requires_grad = False?
    
    return simclr
'''

def get_simclr_based_model() -> nn.Module:
    model = SimCLRBasedModel()
    print(f'loaded simclr based model:{model}')
    return model

class ResNetEncoder(torch_models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""
    def __init__(self, block, layers):
        super().__init__(block, layers)  
        self.fc = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.fc:
            x = self.fc(x)

        return x

class ResNet50(ResNetEncoder):
    def __init__(self):
        super().__init__(torch_models.resnet.Bottleneck, [3, 4, 6, 3])



class EncodeProject(nn.Module):
    def __init__(self):
        super().__init__()

        self.convnet = ResNet50()
        print(f'** Loading pretrained  weights **')
        state_dict = torch.utils.model_zoo.load_url(model_urls['ResNet50'.lower()])
        self.convnet.load_state_dict(state_dict)
        self.encoder_dim = 2048

        num_params = sum(p.numel() for p in self.convnet.parameters() if p.requires_grad)

        print(f'======> Encoder: output dim {self.encoder_dim} | {num_params/1e6:.3f}M parameters')

        self.proj_dim = 128
        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, 128, bias=False)),
            ('bn2', BatchNorm1dNoBias(128)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))

    def forward(self, x, out='h'):
        h = self.convnet(x)
        if out == 'h':
            return h
        return self.projection(h)

    @classmethod
    def load(cls, ckpt):    
        res = cls()
        res.load_state_dict(ckpt['state_dict'])
        return res

class SimCLRBasedModel(EncodeProject):
    def __init__(self) -> None:
        super().__init__()
        ckpt = torch.load('./checkpoints/resnet50_imagenet_bs2k_epochs200.pth.tar')
        self.encoder = EncodeProject.load(ckpt)
        self.fc = generate_xception_head_mlp() # TODO change this maybe

    def encode(self, x, out='h'):
        return self.model(x, out=out)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc(h)
    

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False