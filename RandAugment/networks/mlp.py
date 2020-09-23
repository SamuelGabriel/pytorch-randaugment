import torch
from torch import nn


def MLP(D_out,in_dims):
    in_dim = 1
    for d in in_dims: in_dim *= d
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, 200),
        nn.Tanh(),
        nn.Linear(200,200),
        nn.Tanh(),
        nn.Linear(200,D_out)
    )