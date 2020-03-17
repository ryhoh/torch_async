import torchvision.models as models
import torch.nn as nn

# 準同期式レイヤ
from layers.static import OptimizedSemiSyncLinear


class InceptionV3(nn.Module):
    def __init__(self,
                 pooling="average", num_classes=100, poolingshape=1,
                 middleshape=4096, sync="normal"):
        super(InceptionV3, self).__init__()
        """ use inception_v3"""
        inceptionv3 = models.inception_v3(pretrained=False)
        """ TODO: inceptionはpooling層なし"""

        """ apply semi-sync"""
        in_shape = 2048
        if sync == "normal":
            inceptionv3.fc = nn.Sequential(
                nn.Linear(in_shape, middleshape),
                nn.ReLU(middleshape),
                nn.Linear(middleshape, num_classes)
            )
        elif sync == "semi":
            inceptionv3.fc = nn.Sequential(
                OptimizedSemiSyncLinear(nn.Linear(in_shape, middleshape)),
                nn.ReLU(middleshape),
                nn.Linear(middleshape, num_classes),
            )
        elif sync == "none":
            inceptionv3.fc = nn.Sequential(
                nn.Linear(in_shape, num_classes),
            )
        else:
            raise ValueError("syncの値が不正です")

        self.incep_v3 = inceptionv3

    def forward(self, x):
        x = self.incep_v3(x)
        return x
