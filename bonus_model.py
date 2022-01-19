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
    #model = get_simclr_based_model()
    # model = Resnet30BasedModel()
    model = Discriminator()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('./checkpoints/bonus_model.pt')['model'])
    return model

def get_simclr_based_model() -> nn.Module:
    model = SimCLRBasedModel()
    # print(f'loaded simclr based model:{model}')
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

class Resnet30BasedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # super().__init__(torch_models.resnet.Bottleneck, [3, 4, 6, 3])
        self.encoder = torch_models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*(list(self.encoder.children())[:-1]))

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        mlp = [
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            # nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        ] 

        self.fc = nn.Sequential(*mlp)

        # for param in self.encoder.fc.parameters():
        #     param.requires_grad = True

        print(self.encoder)

    def forward(self, x):
        h = self.encoder(x).squeeze()
        out = self.fc(h)
        return out
        # return self.encoder(x)

class SimCLRBasedModel(EncodeProject):
    def __init__(self) -> None:
        super().__init__()
        ckpt = torch.load('./checkpoints/resnet50_imagenet_bs2k_epochs200.pth.tar')
        self.encoder = EncodeProject.load(ckpt)
        self.fc = generate_xception_head_mlp()

    def encode(self, x, out='h'):
        return self.model(x, out=out)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc(h)
    
class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

# Code here is based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

nc = 3 # number of channels in image
nz = 1000 # size of z vector
ngf = 256 # num of feature maps in generator
ndf = 64 # num of feature maps in discriminator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 64 x 64
            nn.ConvTranspose2d(ngf, ngf / 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 128 x 128
            nn.ConvTranspose2d(ngf / 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 32, 2048, 4, 2, 0, bias=False),
            # nn.Sigmoid()
        )
        mlp = [
            nn.Linear(2048, 1000),
            nn.Linear(1000, 512),
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            # nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        ]
        self.fc = nn.Sequential(*mlp)

        '''
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        self.lr1 = nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 128 x 128
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        self.bn2 = nn.BatchNorm2d(ndf * 2),
        self.lr2 = nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 32 x 32
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        self.bn3 = nn.BatchNorm2d(ndf * 4),
        self.lr3 = nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 16 x 16
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        self.bn4 = nn.BatchNorm2d(ndf * 8),
        self.lr4 = nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 8 x 8
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
        self.bn5 = nn.BatchNorm2d(ndf * 16),
        self.lr5 =  nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*16) x 4 x 4

        # self.conv6 = nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        # self.sigmoid = nn.Sigmoid()
        '''

    def forward(self, input):
        h = self.main(input).squeeze()
        out = self.fc(h)
        return out
        # return self.fc(self.main(input)).squeeze()

class DCGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discriminator = Discriminator()
        self.discriminator.apply(self.weights_init)

        self.generator = Generator()
        self.generator.apply(self.weights_init)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)