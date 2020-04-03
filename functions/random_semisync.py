import torch
from torch.autograd import Function
from torch.nn.functional import linear
from torch import Tensor


class RandomGroupLinearFunction(Function):
    @staticmethod
    def forward(ctx, *args) -> Tensor:
        x, w, b, p = args
        p_tensor = torch.as_tensor(p).requires_grad_(False)

        ctx.save_for_backward(x, w, b, p_tensor)
        return linear(x, w, b)

    @staticmethod
    def backward(ctx, *args: Tensor) -> tuple:
        grad_output = args[0]
        x, w, b, p_tensor = ctx.saved_tensors
        w = w.t()
        p = float(p_tensor)

        grad_ = grad_output.clone()
        if grad_.is_cuda:
            mask = torch.zeros(grad_.shape, dtype=torch.float32, device='cuda')
        else:
            mask = torch.zeros(grad_.shape, dtype=torch.float32, device='cpu')
        # mask[:, learn_l:learn_r] = 1.0
        mask[:, torch.rand(grad_.shape[1]) < p] = 1.0
        grad_ *= mask

        d_b = torch.sum(grad_, dim=0)
        d_w = torch.matmul(torch.t(x), grad_)
        d_x = torch.matmul(grad_output, torch.t(w))

        return d_x, d_w.t(), d_b, None


class RandomSequentialFunction(Function):
    # Linear と同じ
    @staticmethod
    def forward(ctx, *args) -> Tensor:
        x, w, b = args
        ctx.save_for_backward(x, w, b)
        return linear(x, w, b)

    @staticmethod
    def backward(ctx, *args: Tensor) -> tuple:
        grad_output = args[0]
        x, w, b = ctx.saved_tensors
        w = w.t()

        grad_ = grad_output.clone()
        if grad_.is_cuda:
            mask = torch.zeros(grad_.shape, dtype=torch.float32, device='cuda')
        else:
            mask = torch.zeros(grad_.shape, dtype=torch.float32, device='cpu')
        # mask[:, learn_l:learn_r] = 1.0
        idx = torch.floor(torch.rand(1) * grad_.shape[1])
        assert 0 <= idx.item() < grad_.shape[1]
        mask[:, idx] = 1.0
        grad_ *= mask

        d_b = torch.sum(grad_, dim=0)
        d_w = torch.matmul(torch.t(x), grad_)
        d_x = torch.matmul(grad_output, torch.t(w))

        return d_x, d_w.t(), d_b
