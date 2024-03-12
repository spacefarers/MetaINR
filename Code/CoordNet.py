import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import init
import time
import math


class Sine(nn.Module):
    def __init(self):
        super(Sine,self).__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features) 
            #self.linear.weight.normal_(0,0.05) 
        
    def forward(self, input):
        return self.linear(input)
    
class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.init_weights()

    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features) 
            #self.linear.weight.normal_(0,0.05) 
        
    def forward(self, input):
        return self.act(self.linear(input))

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features,nonlinearity='relu'):
        super(ResBlock,self).__init__()
        nls_and_inits = {'sine':Sine(),
                         'relu':nn.ReLU(inplace=True),
                         'sigmoid':nn.Sigmoid(),
                         'tanh':nn.Tanh(),
                         'selu':nn.SELU(inplace=True),
                         'softplus':nn.Softplus(),
                         'elu':nn.ELU(inplace=True)}

        self.nl = nls_and_inits[nonlinearity]

        self.net = []

        self.net.append(SineLayer(in_features,out_features))

        self.net.append(SineLayer(out_features,out_features))

        self.flag = (in_features!=out_features)

        if self.flag:
            self.transform = SineLayer(in_features,out_features)

        self.net = nn.Sequential(*self.net)
    
    def forward(self,features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)

class CoordNet(nn.Module):
    #A fully connected neural network that also allows swapping out the weights when used with a hypernetwork. Can be used just as a normal neural network though, as well.

    def __init__(self, in_features, out_features, init_features=64,num_res = 10):
        super(CoordNet,self).__init__()

        self.num_res = num_res

        self.net = []

        self.net.append(ResBlock(in_features,init_features))
    
        self.net.append(ResBlock(init_features,2*init_features))

        self.net.append(ResBlock(2*init_features,4*init_features))


        for i in range(self.num_res):
            self.net.append(ResBlock(4*init_features,4*init_features))

        # self.net.append(ResBlock(4*init_features, out_features))
        self.net.append(LinearLayer(4*init_features, out_features))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output
    
class SIREN(nn.Module):
    #A fully connected neural network that also allows swapping out the weights when used with a hypernetwork. Can be used just as a normal neural network though, as well.

    def __init__(self, in_features, out_features, init_features=64,num_res = 10):
        super(SIREN,self).__init__()

        self.num_res = num_res

        self.net = []

        self.net.append(SineLayer(in_features,init_features))
    
        self.net.append(SineLayer(init_features,2*init_features))

        self.net.append(SineLayer(2*init_features,4*init_features))


        for i in range(self.num_res):
            self.net.append(SineLayer(4*init_features,4*init_features))

        # self.net.append(ResBlock(4*init_features, out_features))
        self.net.append(LinearLayer(4*init_features, out_features))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


class Column(nn.Module):
    def __init__(self, in_features=3, num_res=3):
        super(Column,self).__init__()
        self.meta = SIREN(in_features=in_features, out_features=1, init_features=64, num_res=3)
        self.trained=False
        self.requires_grad_(False)

    def forward(self, coords):
        tmp_net = self.meta.net[:-1]
        tmp_net = nn.Sequential(*tmp_net)
        output = tmp_net(coords)
        return output

class Linear(nn.Module):
    def __init__(self, column:Column, in_features=256, num_res=1):
        super(Linear,self).__init__()
        self.column = column
        self.net = []

        for i in range(num_res):
            self.net.append(SineLayer(in_features, in_features))
        self.net.append(LinearLayer(in_features,1))

        self.net = nn.Sequential(*self.net)
        if isinstance(self.column.meta.net[-1], LinearLayer):
            # copy the weights from the meta network
            self.net[-1].linear.weight.data = self.column.meta.net[-1].linear.weight.data

    def forward(self, coords):
        coords = self.column(coords)
        output = self.net(coords)
        return output