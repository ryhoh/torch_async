import torch
import torch.nn as nn
from torchvision.models import vgg as vggmodel

# 準同期式レイヤ
from layers.static import OptimizedSemiSyncLinear


class Stripe3_semi_VGG16(nn.Module):
    def __init__(self,
                 num_classes=100, model_type="vgg_without_maxpool",
                 pooling="average", poolingshape=7, middleshape=4096,
                 dropout_prob=0.5, bnflag=True, first_async=True,
                 second_async=True, third_async=True):
        super(Stripe3_semi_VGG16, self).__init__()

        cfgs = {
            'vgg_with_maxpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg_without_maxpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
        }

        """ vgg16 """
        vgg = vggmodel.VGG(vggmodel.make_layers(cfgs[model_type], batch_norm=bnflag))
        self.vgg = nn.Sequential(*list(vgg.children())[:-2])

        """ global average pooling or Max Pooling """
        if pooling == "average":
            self.pool = nn.AdaptiveAvgPool2d((poolingshape, poolingshape))
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((poolingshape, poolingshape))
        else:
            raise ValueError("poolingの値が不正です")

        """ FCを定義"""
        in_shape = 512 * poolingshape * poolingshape

        if first_async:
            fc = [
                OptimizedSemiSyncLinear(nn.Linear(in_shape, middleshape)),
                nn.ReLU(middleshape),
                nn.Dropout(p=dropout_prob)
            ]
        else:
            fc = [
                nn.Linear(in_shape, middleshape),
                nn.ReLU(middleshape),
                nn.Dropout(p=dropout_prob)
            ]
        if second_async:
            fc += [
                OptimizedSemiSyncLinear(nn.Linear(middleshape, middleshape)),
                nn.ReLU(middleshape),
                nn.Dropout(p=dropout_prob)
            ]
        else:
            fc += [
                nn.Linear(middleshape, middleshape),
                nn.ReLU(middleshape),
                nn.Dropout(p=dropout_prob)
            ]
        if third_async:
            fc += [
                OptimizedSemiSyncLinear(nn.Linear(middleshape, middleshape)),
                nn.ReLU(middleshape),
                nn.Dropout(p=dropout_prob)
            ]
        else:
            fc += [
                nn.Linear(middleshape, middleshape),
                nn.ReLU(middleshape),
                nn.Dropout(p=dropout_prob)
            ]
        fc.append(nn.Linear(middleshape, num_classes))

        self.fc = nn.Sequential()
        for i, layer in enumerate(fc):
            self.fc.add_module(str(i), layer)

    def forward(self, x):
        x = self.vgg(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
