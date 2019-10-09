from copy import deepcopy

from math import sqrt

from torch import Tensor
from torch.nn.modules import *

from functions.static import StaticGroupLinearFunction, OptimizedGroupLinearFunction, OptimizedContinuousLinearFunction


class Rotatable(object):
    def rotate(self):
        raise NotImplementedError


class SemiSyncLinear(Linear, Rotatable):
    def __init__(self, linear: Linear, group_list: list = None):
        super().__init__(linear.in_features, linear.out_features, bias=True)
        # パラメータを引き継ぎつつ、準同期式更新に対応させる
        self.weight = deepcopy(linear.weight)
        self.bias = deepcopy(linear.bias)
        self.learn_l = None
        self.learn_r = None

        output_features = linear.out_features

        if group_list is not None:
            group_list_copy = group_list.copy()
            nueron_sum = sum(group_list_copy)
            if nueron_sum > output_features:
                raise ValueError("存在するニューロン数より多い割当")
        else:
            # サイズ，グループ数ともに sqrt(output_features) にする
            num = int(sqrt(output_features))
            group_list_copy = [num for _ in range(num)]
            nueron_sum = sum(group_list_copy)

        if nueron_sum < output_features:  # 割り当てられていないニューロンでもう1グループ作成
            group_list_copy.append(output_features - nueron_sum)

        # グループに属するニューロンを高速にスライスするために累積和を用いる
        group_delim = [0] + group_list_copy
        for i in range(1, len(group_delim)):
            group_delim[i] += group_delim[i - 1]

        group_i = 1

        self.group_delim = group_delim
        self.group_i = group_i
        # self.register_buffer("group_delim", torch.as_tensor(group_delim).requires_grad_(False))
        # self.register_buffer("group_i",     torch.as_tensor(group_i).requires_grad_(False))

    def forward(self, input_tensor) -> Tensor:
        self.learn_l = self.group_delim[self.group_i-1]
        self.learn_r = self.group_delim[self.group_i]

        matmul = StaticGroupLinearFunction.apply
        res = matmul(input_tensor, self.weight, self.bias, self.learn_l, self.learn_r)

        return res

    def rotate(self):
        self.group_i += 1
        if self.group_i == len(self.group_delim):
            self.group_i = 1  # 最初のグループへ


class ContinuousLinear(SemiSyncLinear):
    def __init__(self, linear: Linear):
        super(ContinuousLinear, self)\
            .__init__(linear, [1 for _ in range(linear.out_features)])


class OptimizedSemiSyncLinear(Linear, Rotatable):
    def __init__(self, linear: Linear, group_list: list = None):
        super().__init__(linear.in_features, linear.out_features, bias=True)
        # パラメータを引き継ぎつつ、準同期式更新に対応させる
        self.weight = deepcopy(linear.weight)
        self.bias = deepcopy(linear.bias)
        self.learn_l = None
        self.learn_r = None

        output_features = linear.out_features

        if group_list is not None:
            group_list_copy = group_list.copy()
            nueron_sum = sum(group_list_copy)
            if nueron_sum > output_features:
                raise ValueError("存在するニューロン数より多い割当")
        else:
            # サイズ，グループ数ともに sqrt(output_features) にする
            num = int(sqrt(output_features))
            group_list_copy = [num for _ in range(num)]
            nueron_sum = sum(group_list_copy)

        if nueron_sum < output_features:  # 割り当てられていないニューロンでもう1グループ作成
            group_list_copy.append(output_features - nueron_sum)

        # グループに属するニューロンを高速にスライスするために累積和を用いる
        group_delim = [0] + group_list_copy
        for i in range(1, len(group_delim)):
            group_delim[i] += group_delim[i - 1]

        group_i = 1

        self.group_delim = group_delim
        self.group_i = group_i

    def forward(self, input_tensor) -> Tensor:
        self.learn_l = self.group_delim[self.group_i-1]
        self.learn_r = self.group_delim[self.group_i]

        matmul = OptimizedGroupLinearFunction.apply
        res = matmul(input_tensor, self.weight, self.bias, self.learn_l, self.learn_r)

        return res

    def rotate(self):
        self.group_i += 1
        if self.group_i == len(self.group_delim):
            self.group_i = 1  # 最初のグループへ


class OptimizedContinuousLinear(Linear, Rotatable):
    def __init__(self, linear: Linear):
        super().__init__(linear.in_features, linear.out_features, bias=True)
        # パラメータを引き継ぎつつ、準同期式更新に対応させる
        self.weight = deepcopy(linear.weight)
        self.bias = deepcopy(linear.bias)

        self.learn_idx = 0  # 学習ニューロンのインデックス
        self.neuron_n = self.bias.shape[0]

    def forward(self, input_tensor) -> Tensor:
        matmul = OptimizedContinuousLinearFunction.apply
        res = matmul(input_tensor, self.weight, self.bias, self.learn_idx)

        return res

    def rotate(self):
        self.learn_idx += 1
        if self.learn_idx == self.neuron_n:
            self.learn_idx = 0  # 最初のニューロンへ
