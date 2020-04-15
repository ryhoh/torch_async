import torch
from torch.autograd import Function
from torch.nn.functional import linear
from torch import Tensor


class GroupLinearFunction(Function):
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
        w = w.t()
        learn_l, learn_r = int(learn_l), int(learn_r)

        # バイアスへの勾配は、0ベクトルを作って必要な要素だけ値を入れる
        if grad.is_cuda:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32, device='cuda')
        else:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32)
        d_b[learn_l:learn_r] = torch.sum(grad[:, learn_l:learn_r], dim=0)

        # パラメータへの勾配は、0行列を作って必要な行だけ値を入れる
        if grad.is_cuda:
            d_w = torch.zeros(size=(x.shape[1], grad.shape[1]), dtype=torch.float32, device='cuda')
        else:
            d_w = torch.zeros(size=(x.shape[1], grad.shape[1]), dtype=torch.float32)
        d_w[:, learn_l:learn_r] = torch.matmul(x.t(), grad[:, learn_l:learn_r])

        d_x = torch.matmul(grad, torch.t(w))

        return d_x, d_w.t(), d_b, None, None


class SequentialLinearFunction(Function):
    @staticmethod
    def forward(ctx, *args) -> Tensor:
        x, w, b, learn_idx = args
        learn_idx = torch.as_tensor(learn_idx).requires_grad_(False)

        ctx.save_for_backward(x, w, b, learn_idx)
        return linear(x, w, b)

    @staticmethod
    def backward(ctx, *args: Tensor) -> tuple:
        grad = args[0]
        x, w, b, learn_idx = ctx.saved_tensors
        w = w.t()
        learn_idx = int(learn_idx)

        # バイアスへの勾配は、0ベクトルを作って1要素だけ値を入れる
        if grad.is_cuda:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32, device='cuda')
        else:
            d_b = torch.zeros(size=(grad.shape[1],), dtype=torch.float32)
        d_b[learn_idx] = torch.sum(grad[:, learn_idx])

        # パラメータへの勾配は、0行列を作って1行だけ値を入れる
        if grad.is_cuda:
            d_w = torch.zeros(size=(x.shape[1], grad.shape[1]), dtype=torch.float32, device='cuda')
        else:
            d_w = torch.zeros(size=(x.shape[1], grad.shape[1]), dtype=torch.float32)
        d_w[:, learn_idx] = torch.matmul(x.t(), grad[:, learn_idx])

        d_x = torch.matmul(grad, torch.t(w))

        return d_x, d_w.t(), d_b, None
