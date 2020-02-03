import torchvision.models as models
import torch
import torch.nn as nn

# 準同期式レイヤ
from layers.static import OptimizedSemiSyncLinear


class ResNetForImageNet(nn.Module):
    def __init__(self, pooling="average", sync="normal"):
        super(ResNetForImageNet, self).__init__()
        """ use resnet18 """
        resnet = models.resnet18(pretrained=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        """ global average pooling or Max Pooling """
        if pooling == "average":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=7)
        else:
            raise ValueError("poolingの値が不正です")

        """ apply semi-sync"""
        if sync == "normal":
            self.fc = nn.Sequential(
                nn.Linear(512 * 1, 128),
                nn.ReLU(128),
                nn.Linear(128, 1000),
            )
        elif sync == "semi":
            self.fc = nn.Sequential(
                OptimizedSemiSyncLinear(nn.Linear(512 * 1, 128)),
                nn.ReLU(128),
                nn.Linear(128, 1000),
            )
        elif sync == "none":
            self.fc = nn.Sequential(
                nn.Linear(512 * 1, 1000),
            )
        else:
            raise ValueError("syncの値が不正です")

    def forward(self, x):
        x = self.resnet(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


""" Usage
from torchsummary import summary

model = ResNetForImageNet(pooling="max", sync="semi")
summary(model.cuda(), (3, 224, 224))
"""
