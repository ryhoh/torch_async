import torch
import torch.nn as nn

# 準同期式レイヤ
from layers.static import OptimizedSemiSyncLinear


def make_layers(cfg, batch_norm=False):
    """ vggのfeaturesレイヤ生成(pytorchから引用) """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


""" vgg16のアーキテクチャ(pytorchから引用)

注意: 最後のMaxPoolingを抜いている
"""
cfgs = {
    'vgg16_with_maxpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                           512, 512, 512, 'M', 512, 512, 512],
    'vgg16_without_maxpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                              512, 512, 512, 'M', 512, 512, 512, 'M']
}


class VGGForImageNet(nn.Module):
    def __init__(self, model_type='vgg16_without_maxpool', num_classes=1000, pooling="average", sync="normal"):
        super(VGGForImageNet, self).__init__()
        """ vgg16 """
        self.features = make_layers(cfgs[model_type], batch_norm=False)

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
                nn.Linear(128, num_classes),
            )
        elif sync == "semi":
            self.fc = nn.Sequential(
                OptimizedSemiSyncLinear(nn.Linear(512 * 1, 128)),
                nn.ReLU(128),
                nn.Linear(128, num_classes),
            )
        elif sync == "none":
            self.fc = nn.Sequential(
                nn.Linear(512 * 1, 1000),
            )
        else:
            raise ValueError("syncの値が不正です")

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


""" Usage
from torchsummary import summary

model = VGGForImageNet()
summary(model.cuda(), (3, 224, 224))
"""
