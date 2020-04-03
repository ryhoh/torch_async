from copy import deepcopy

from torch.nn.modules import Conv2d


# 更新するグループを交代することが可能なクラスの基底クラス
class Rotatable(object):
    def rotate(self):
        raise NotImplementedError


class MyConv2d(Conv2d):
    def __init__(self, conv2d: Conv2d):
        has_bias = False if conv2d.bias is None else True
        super(MyConv2d, self).__init__(
            conv2d.in_channels, conv2d.out_channels, conv2d.kernel_size,
            conv2d.stride, conv2d.padding, conv2d.dilation, conv2d.groups,
            has_bias, conv2d.padding_mode)
        # パラメータを引き継ぎつつ、準同期式更新に対応させる
        self.weight = deepcopy(conv2d.weight)
        if conv2d.bias is not None:
            self.bias = deepcopy(conv2d.bias)

