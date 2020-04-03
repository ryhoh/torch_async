from copy import deepcopy

import numpy as np
from torch import Tensor
from torch.nn.modules import *

from functions.stochastic_semisync import StochasticGroupLinearFunction


# 各ニューロン：ランダムにグループ分割（飛び跳びになってもよい）しランダムな順に更新
# グループサイズはできるだけ均等
class StochasticSemiSyncLinear(Linear):
    def __init__(self, linear: Linear, group_size: int):
        super().__init__(linear.in_features, linear.out_features, bias=True)
        # パラメータを引き継ぎつつ、準同期式更新に対応させる
        self.weight = deepcopy(linear.weight)
        self.bias = deepcopy(linear.bias)

        if group_size <= 0 or linear.out_features < group_size:
            raise ValueError("illegal group_size", group_size)
        self.group_size = group_size
        self.group_idx = np.ndarray([i % group_size for i in range(linear.out_features)])
        self.group_cur = 0  # 次に更新するグループは何番目？

    def forward(self, input_tensor) -> Tensor:
        if self.group_cur == self.group_size:
            self.group_cur = 0
            self.shuffle()

        matmul = RandomGroupLinearFunction.apply
        # res = matmul(input_tensor, self.weight, self.bias, self.learn_l, self.learn_r)
        res = matmul(input_tensor, self.weight, self.bias, self.group_idx, self.group_cur)
        self.group_cur += 1

        return res

    def shuffle(self):
        np.random.shuffle(self.group_idx)


# レイヤ内を1ニューロンずつに分割，順番はランダム
class StochasticSequentialLinear(StochasticSemiSyncLinear):
    def __init__(self, linear: StochasticSemiSyncLinear):
        super().__init__(linear, linear.out_features)
