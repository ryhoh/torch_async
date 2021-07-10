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
        if w.ndim == 2:  # Primitive Backward
            w = w.t()
        elif w.ndim == 3:  # 3D Backward (not sure...)
            print(w.shape)
            w = w.permute(0, 2, 1)

        learn_l, learn_r = int(learn_l), int(learn_r)

        # バイアスへの勾配は、0ベクトルを作って必要な要素だけ値を入れる
        # gradients for bias, make 0 vector and insert value into needed element
        if grad.is_cuda:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32, device='cuda')
        else:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32)
        d_b[learn_l:learn_r] = torch.sum(grad[:, learn_l:learn_r], dim=0)

        # 重みへの勾配は、0行列を作って必要な行だけ値を入れる
        # gradients for weights, make 0 matrix and insert value into needed column
        if grad.is_cuda:
            d_w = torch.zeros(size=(x.shape[1], grad.shape[1]), dtype=torch.float32, device='cuda')
        else:
            d_w = torch.zeros(size=(x.shape[1], grad.shape[1]), dtype=torch.float32)

        if x.ndim == 2:  # Primitive Backward
            d_w[:, learn_l:learn_r] = torch.matmul(x.t(), grad[:, learn_l:learn_r])
        elif x.ndim == 3:  # 3D Backward (not sure...)
            print(d_w.shape, x.shape, grad.shape)
            d_w[:, learn_l:learn_r] = torch.matmul(x.permute(0, 2, 1), grad[:, learn_l:learn_r])
        else:
            raise BackwardError("x.ndim == %d" % x.ndim)

        if w.ndim == 2:  # Primitive Backward
            d_x = torch.matmul(grad, torch.t(w))
        elif w.ndim == 3:  # 3D Backward (not sure...)
            d_x = torch.matmul(grad, w.permute(0, 2, 1))
        else:
            raise BackwardError("w.ndim == %d" % w.ndim)

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
