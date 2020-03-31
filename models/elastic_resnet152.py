import torchvision.models as models
import torch
import torch.nn as nn

# 準同期式レイヤ
from layers.static import OptimizedSemiSyncLinear


class Elastic_ResNet152(nn.Module):
    def __init__(self,
                 num_classes=100,
                 pooling="average", poolingshape=1, middleshape=4096,
                 sync="normal", dropout_prob=0.5, deepness=2):
        super(Elastic_ResNet152, self).__init__()
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

        """ FCを定義"""
        in_shape = 2048 * poolingshape * poolingshape
        if sync == "normal":
            fc = [
                nn.Linear(in_shape, middleshape),
                nn.ReLU(middleshape),
                nn.Dropout(p=dropout_prob)
            ]
            if deepness > 1:
                for _ in range(deepness - 1):
                    fc += [
                        nn.Linear(middleshape, middleshape),
                        nn.ReLU(middleshape),
                        nn.Dropout(p=dropout_prob)
                    ]
            fc.append(nn.Linear(middleshape, num_classes))
        elif sync == "semi":
            fc = [
                OptimizedSemiSyncLinear(nn.Linear(in_shape, middleshape)),
                nn.ReLU(middleshape),
                nn.Dropout(p=dropout_prob)
            ]
            if deepness > 1:
                for _ in range(deepness - 1):
                    fc += [
                        OptimizedSemiSyncLinear(nn.Linear(middleshape, middleshape)),
                        nn.ReLU(middleshape),
                        nn.Dropout(p=dropout_prob)
                    ]
            fc.append(nn.Linear(middleshape, num_classes))
        elif sync == "none":
            fc = nn.Sequential(
                nn.Linear(in_shape, num_classes),
            )
        else:
            raise ValueError("引数fcの値が不正です")

        self.fc = nn.Sequential()
        for i, layer in enumerate(fc):
            self.fc.add_module(str(i), layer)

    def forward(self, x):
        x = self.resnet(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
