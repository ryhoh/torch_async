from copy import deepcopy

from torch import Tensor
from torch.nn.modules import *

from .base import MyConv2d
from functions import RandomGroupLinearFunction, RandomSequentialFunction
from functions import RandomConv2dSemiSyncFuncion


# 各ニューロン：確率 p で更新
class RandomSemiSyncLinear(Linear):
    def __init__(self, linear: Linear, p: float = 0.5):
        super().__init__(linear.in_features, linear.out_features, bias=True)
        # パラメータを引き継ぎつつ、準同期式更新に対応させる
        self.weight = deepcopy(linear.weight)
        self.bias = deepcopy(linear.bias)

        if p < 0 or 1 < p:
            raise ValueError("p must be 0~1 but given", p)
        self.p = p

    def forward(self, input_tensor) -> Tensor:
        matmul = RandomGroupLinearFunction.apply
        # res = matmul(input_tensor, self.weight, self.bias, self.learn_l, self.learn_r)
        res = matmul(input_tensor, self.weight, self.bias, self.p)

        return res


# レイヤ内で1つだけ更新（どれが選ばれるかはランダム）
class RandomSequentialLinear(Linear):
    def __init__(self, linear: Linear):
        super().__init__(linear.in_features, linear.out_features, bias=True)
        # パラメータを引き継ぎつつ、準同期式更新に対応させる
        self.weight = deepcopy(linear.weight)
        self.bias = deepcopy(linear.bias)

    def forward(self, input_tensor) -> Tensor:
        matmul = RandomSequentialFunction.apply
        # res = matmul(input_tensor, self.weight, self.bias, self.learn_l, self.learn_r)
        res = matmul(input_tensor, self.weight, self.bias)

        return res


class RandomSemiSyncConv2d(MyConv2d):
    def __init__(self, conv2d: Conv2d, p: float = 0.5):
        super(RandomSemiSyncConv2d, self).__init__(conv2d)
        if p < 0 or 1 < p:
            raise ValueError("p must be 0~1 but given", p)
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        conv = RandomConv2dSemiSyncFuncion.apply
        res = conv(input, self.weight, self.bias, self.stride,
                   self.padding, self.dilation, self.groups, self.p)
        return res
