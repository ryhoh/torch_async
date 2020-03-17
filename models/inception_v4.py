import torch.nn as nn

from models.inception_v4_original import inception_v4
# 準同期式レイヤ
from layers.static import OptimizedSemiSyncLinear


class InceptionV4(nn.Module):
    def __init__(self,
                 pooling="average", num_classes=100, poolingshape=1,
                 middleshape=4096, sync="normal"):
        super(InceptionV4, self).__init__()
        """ use inception v4"""
        inceptionv4 = inception_v4()
        self.last_linear = nn.Sequential(*list(inceptionv4.children())[:-1])
        """ TODO: inceptionはpooling層なし"""

        """ apply semi-sync"""
        in_shape = 1536
        if sync == "normal":
            inceptionv4.last_linear = nn.Sequential(
                nn.Linear(in_shape, middleshape),
                nn.ReLU(middleshape),
                nn.Linear(middleshape, num_classes)
            )
        elif sync == "semi":
            inceptionv4.last_linear = nn.Sequential(
                OptimizedSemiSyncLinear(nn.Linear(in_shape, middleshape)),
                nn.ReLU(middleshape),
                nn.Linear(middleshape, num_classes),
            )
        elif sync == "none":
            inceptionv4.last_linear = nn.Sequential(
                nn.Linear(in_shape, num_classes),
            )
        else:
            raise ValueError("syncの値が不正です")

        self.incep_v4 = inceptionv4

    def forward(self, x):
        x = self.incep_v4(x)

        return x
