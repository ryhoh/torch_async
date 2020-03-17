import torchvision.models as models
import torch
import torch.nn as nn

# 準同期式レイヤ
from layers.static import OptimizedSemiSyncLinear


class ResNet152(nn.Module):
    def __init__(self,
                 pooling="average", num_classes=100, poolingshape=1,
                 middleshape=2048, sync="normal"):
        super(ResNet152, self).__init__()
        """ use resnet152 """
        resnet = models.resnet152(pretrained=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        """ global average pooling or Max Pooling """
        if pooling == "average":
            self.pool = nn.AdaptiveAvgPool2d((poolingshape, poolingshape))
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((poolingshape, poolingshape))
        else:
            raise ValueError("poolingの値が不正です")

        """ apply semi-sync"""
        in_shape = 2048 * poolingshape * poolingshape
        if sync == "normal":
            self.fc = nn.Sequential(
                nn.Linear(in_shape, middleshape),
                nn.ReLU(middleshape),
                nn.Linear(middleshape, num_classes),
            )
        elif sync == "semi":
            self.fc = nn.Sequential(
                OptimizedSemiSyncLinear(nn.Linear(in_shape, middleshape)),
                nn.ReLU(middleshape),
                nn.Linear(middleshape, num_classes),
            )
        elif sync == "none":
            self.fc = nn.Sequential(
                nn.Linear(in_shape, num_classes),
            )
        else:
            raise ValueError("syncの値が不正です")

    def forward(self, x):
        x = self.resnet(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
