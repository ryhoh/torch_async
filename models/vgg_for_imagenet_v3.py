import torch
import torch.nn as nn
from torchvision.models import vgg as vggmodel

# 準同期式レイヤ
from layers.static import OptimizedSemiSyncLinear


class VGGForImageNet(nn.Module):
    def __init__(self, num_classes=100, model_type="vgg_without_maxpool", pooling="average", poolingshape=1, middleshape=4096, sync="normal"):
        super(VGGForImageNet, self).__init__()

        cfgs = {
            'vgg_with_maxpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg_without_maxpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
        }

        """ vgg16 """
        vgg = vggmodel.VGG(vggmodel.make_layers(cfgs[model_type], batch_norm=False))
        self.vgg = nn.Sequential(*list(vgg.children())[:-2])

        """ global average pooling or Max Pooling """
        if pooling == "average":
            self.pool = nn.AdaptiveAvgPool2d((poolingshape, poolingshape))
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((poolingshape, poolingshape))
        else:
            raise ValueError("poolingの値が不正です")

        """ apply semi-sync"""
        in_shape = 512 * poolingshape * poolingshape
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
        x = self.vgg(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
