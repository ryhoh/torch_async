import torch
from torch.autograd import Function
from torch.nn.functional import linear
from torch import Tensor


class StochasticGroupLinearFunction(Function):
    @staticmethod
    def forward(ctx, *args) -> Tensor:
        x, w, b, idx, cur = args
        idx_tensor = torch.as_tensor(idx).requires_grad_(False)
        cur_tensor = torch.as_tensor(cur).requires_grad_(False)

        ctx.save_for_backward(x, w, b, idx_tensor, cur_tensor)
        return linear(x, w, b)

    @staticmethod
    def backward(ctx, *args: Tensor) -> tuple:
        grad_output = args[0]
        x, w, b, idx_tensor, cur_tensor = ctx.saved_tensors
        w = w.t()
        # idx_tensor = np.asarray(idx_tensor)
        cur = int(cur_tensor)

        grad_ = grad_output.clone()
        dev = 'cuda' if grad_.is_cuda else 'cpu'
        mask = torch.zeros(grad_.shape, dtype=torch.float32, device=dev)
        mask[idx_tensor == cur] = 1.0
        grad_ *= mask

        d_b = torch.sum(grad_, dim=0)
        d_w = torch.matmul(torch.t(x), grad_)
        d_x = torch.matmul(grad_output, torch.t(w))

        return d_x, d_w.t(), d_b, None, None
