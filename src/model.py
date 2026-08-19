from torchvision import models
from torch import nn


def get_model(num_outputs):
    net = models.resnet34()
    net.fc = nn.Linear(in_features=net.fc.in_features, out_features=num_outputs)
    return net