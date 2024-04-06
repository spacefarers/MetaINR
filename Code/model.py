from torch import nn
from CoordNet import SIREN,SineLayer,LinearLayer
import config

class Base(nn.Module):
    def __init__(self, additional_layers=0):
        super(Base, self).__init__()
        self.net = [
            SineLayer(3, 64),
            SineLayer(64, 128),
            SineLayer(128, 256),
        ]
        for i in range(additional_layers):
            self.net.append(SineLayer(256, 256))
        self.net = nn.Sequential(*self.net)
        # self.net.requires_grad_(False)

    def forward(self, coords):
        return coords

class Backbone(nn.Module):
    def __init__(self, base=None, layers=2):
        super(Backbone, self).__init__()
        self.net = [
            SineLayer(3, 64),
            SineLayer(64, 128),
            SineLayer(128, 256),
        ]
        for i in range(layers):
            self.net.append(SineLayer(256, 256))
        self.net = nn.Sequential(*self.net)
        # self.net.requires_grad_(False)
        self.base = base

    def forward(self, coords):
        return self.net(self.base(coords)) if self.base is not None else self.net(coords)

class Head(nn.Module):
    def __init__(self, backbone=None, layers=1):
        super(Head, self).__init__()
        self.net = []
        for i in range(layers):
            self.net.append(SineLayer(256, 256))
        self.net.append(LinearLayer(256,1))
        self.net = nn.Sequential(*self.net)
        # self.net.requires_grad_(False)
        self.backbone = backbone

    def forward(self, coords=None):
        return self.net(self.backbone(coords) if self.backbone is not None else coords)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear") != -1:
        nn.init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class MetaModel:
    def __init__(self):
        self.backbone = Backbone().to(config.device)
        self.heads = []
        self.frame_head_correspondence = [0]*len(config.test_timesteps)
        self.replay_buffer = []

    def add_head(self):
        self.heads.append(Head())