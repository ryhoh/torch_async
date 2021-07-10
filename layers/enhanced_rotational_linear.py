import sys

from rotational_update import RotationalLinear
import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn.functional import linear


class BackwardError(Exception):
    pass


class EnhancedRotationalLinearFunction(Function):
    @staticmethod
    def forward(ctx, *args) -> Tensor:
        x, w, b, learn_left, learn_right = args
        learn_l = torch.as_tensor(learn_left).requires_grad_(False)
        learn_r = torch.as_tensor(learn_right).requires_grad_(False)

        ctx.save_for_backward(x, w, b, learn_l, learn_r)
        return linear(x, w, b)

    @staticmethod
    def backward(ctx, *args: Tensor) -> tuple:
        grad = args[0]
        x, w, b, learn_l, learn_r = ctx.saved_tensors

        # torch.Size([N, 577, 768]) torch.Size([768, 768]) torch.Size([768]) torch.Size([N, 577, 768])
        print(x.shape, w.shape, b.shape, grad.shape)

        w = w.t()
        learn_l, learn_r = int(learn_l), int(learn_r)

        # バイアスへの勾配は、0ベクトルを作って必要な要素だけ値を入れる
        # gradients for bias, make 0 vector and insert value into needed element
        if grad.is_cuda:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32, device='cuda')
        else:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32)

        d_b[learn_l:learn_r] = torch.sum(grad[:, learn_l:learn_r], dim=(0, 2))
        sys.stderr.write("c\n")

        # 重みへの勾配は、0行列を作って必要な行だけ値を入れる
        # gradients for weights, make 0 matrix and insert value into needed column
        d_w = torch.zeros(
            size=(x.shape[x.ndim - 1], grad.shape[grad.ndim - 1]),
            dtype=torch.float32,
            device='cuda' if grad.is_cuda else 'cpu'
        )

        print(d_w.shape)
        if x.ndim == 2:  # Primitive Backward
            d_w[:, learn_l:learn_r] = torch.matmul(x.t(), grad[:, learn_l:learn_r])
        elif x.ndim == 3:  # 3D Backward (not sure...)
            d_w[:, learn_l:learn_r] = torch.bmm(x.permute(0, 2, 1), grad[:, learn_l:learn_r])
        else:
            raise BackwardError("d_w.ndim == %d" % d_w.ndim)
        sys.stderr.write("d\n")

        d_x = torch.matmul(grad, torch.t(w))
        sys.stderr.write("e\n")

        if d_w.ndim == 2:  # Primitive Backward
            return d_x, d_w.t(), d_b, None, None
        elif w.ndim == 3:  # 3D Backward (not sure...)
            return d_x, d_w.permute(0, 2, 1), d_b, None, None
        else:
            raise BackwardError("d_w.ndim == %d" % d_w.ndim)


class EnhancedRotationalLinear(RotationalLinear):
    def forward(self, input_tensor) -> Tensor:
        """
        Feed-forward method
        Almost same to normal Linear object.
        Save variables for learning group.

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        res : Output of feed-forwarding.
        """
        self.learn_l = self.group_partition[self.group_i - 1]
        self.learn_r = self.group_partition[self.group_i]

        matmul = EnhancedRotationalLinearFunction.apply
        res = matmul(input_tensor, self.weight, self.bias, self.learn_l, self.learn_r)

        return res
