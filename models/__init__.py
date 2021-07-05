from models.pytorch_resnet_cifar10.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from models.densenet_pytorch import densenet
from models.darknet import Darknet53


def densenet12_40():
    return densenet.DenseNet(growthRate=12, depth=40, reduction=1,
                             bottleneck=False, nClasses=10)


def densenet12_100():
    return densenet.DenseNet(growthRate=12, depth=100, reduction=1,
                             bottleneck=False, nClasses=10)


def densenet24_100():
    return densenet.DenseNet(growthRate=24, depth=100, reduction=1,
                             bottleneck=False, nClasses=10)


def densenetBC_12_100():
    return densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                             bottleneck=True, nClasses=10)


def densenetBC_24_250():
    return densenet.DenseNet(growthRate=24, depth=250, reduction=0.5,
                             bottleneck=True, nClasses=10)


def densenetBC_40_190():
    return densenet.DenseNet(growthRate=40, depth=190, reduction=0.5,
                             bottleneck=True, nClasses=10)
