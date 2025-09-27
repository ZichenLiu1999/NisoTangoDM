from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import Kabsch


class MLP(nn.Module):
    def __init__(self, layers, scale=1.0, activation='SiLU'):
        super(MLP, self).__init__()
        self.depth = len(layers) - 1
        self.activation = getattr(torch.nn, activation)
        self.scale = scale
        
        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))
        layer_list.append(
            ("layer_%d" % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layer_dict = OrderedDict(layer_list)   
        self.layers = torch.nn.Sequential(layer_dict)
    
    def forward(self, x, label=None):
        if label is not None:
            label = label.reshape(-1, 1) * self.scale
            state = torch.cat((x, label), dim=1)
        else:
            state = x
        out = self.layers(state)
        return out


class EMLP(nn.Module):
    def __init__(self, layers, xref, scale=1.0, activation='SiLU'):
        super(EMLP, self).__init__()
        self.depth = len(layers) - 1
        self.xref = xref
        self.natom = xref.shape[0]
        self.activation = getattr(torch.nn, activation)
        self.scale = scale
        
        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))
        layer_list.append(
            ("layer_%d" % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layer_dict = OrderedDict(layer_list)
        
        self.layers = torch.nn.Sequential(layer_dict)
    
    def forward(self, x, label):
        x = x.reshape(-1, self.natom, 3)
        # x = x - x.mean(dim=-2, keepdim=True)
        assert x.shape[1] == self.natom and x.shape[2] == 3, 'shape of input tensor is wrong'
        label = label.reshape(-1, 1) * self.scale
        
        R, b = Kabsch(x, self.xref)
        aligned_x = torch.matmul(x - b, R.transpose(1, 2))
        state = torch.cat((torch.flatten(aligned_x, start_dim=1), label), dim=1)
        
        out = torch.matmul(self.layers(state).reshape(x.shape[0], self.natom, 3), R)
        return out.view(-1, 3 * self.natom)


if __name__ == "__main__":
    pass
    
    # timesteps = torch.randn(100)
    # embedding_dim = 11
