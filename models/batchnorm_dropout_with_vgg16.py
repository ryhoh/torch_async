import torch
import torch.nn as nn
from torchvision.models import vgg as vggmodel

# 準同期式レイヤ
from layers.static import OptimizedSemiSyncLinear


class Batchnorm_drouput_with_vgg16(nn.Module):
    def __init__(self,
                 num_classes=100, model_type="vgg_without_maxpool",
                 pooling="average", poolingshape=7, middleshape=4096,
                 sync="normal", dropout_prob=0.5, deepness=2, cnn_bn_flag=True,
                 fc_bn_flag=True, fc_do_flag=True):
        super(Batchnorm_drouput_with_vgg16, self).__init__()

        cfgs = {
            'vgg_with_maxpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg_without_maxpool': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
        }

        """ vgg16 """
        vgg = vggmodel.VGG(vggmodel.make_layers(cfgs[model_type], batch_norm=cnn_bn_flag))
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
        if sync == "normal":
            fc = [nn.Linear(in_shape, middleshape)]
            if fc_bn_flag:
                fc.append(nn.BatchNorm1d(middleshape))
            fc.append(nn.ReLU(middleshape))
            if fc_do_flag:
                fc.append(nn.Dropout(p=dropout_prob))

            if deepness > 1:
                for _ in range(deepness - 1):
                    fc.append(nn.Linear(middleshape, middleshape))
                    if fc_bn_flag:
                        fc.append(nn.BatchNorm1d(middleshape))
                    fc.append(nn.ReLU(middleshape))
                    if fc_do_flag:
                        fc.append(nn.Dropout(p=dropout_prob))
            fc.append(nn.Linear(middleshape, num_classes))

        elif sync == "semi":
            fc = [OptimizedSemiSyncLinear(nn.Linear(in_shape, middleshape))]
            if fc_bn_flag:
                fc.append(nn.BatchNorm1d(middleshape))
            fc.append(nn.ReLU(middleshape))
            if fc_do_flag:
                fc.append(nn.Dropout(p=dropout_prob))

            if deepness > 1:
                for _ in range(deepness - 1):
                    fc.append(OptimizedSemiSyncLinear(nn.Linear(middleshape, middleshape)))
                    if fc_bn_flag:
                        fc.append(nn.BatchNorm1d(middleshape))
                    fc.append(nn.ReLU(middleshape))
                    if fc_do_flag:
                        fc.append(nn.Dropout(p=dropout_prob))
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
        x = self.vgg(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
