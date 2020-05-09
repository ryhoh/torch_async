import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import Tensor


class RandomConv2dSemiSyncFuncion(Function):
    @staticmethod
    def forward(ctx, *args) -> Tensor:
        r"""
        :param ctx:
        :param args: input, weight, bias, stride, padding, dilation, groups, p
            input: (N C H W)
        :return: convolution processed output
        """
        input, weight, bias, stride, padding, dilation, groups, p = args

        p_tensor = torch.as_tensor(p).requires_grad_(False)
        ctx.save_for_backward(input, weight, bias, p_tensor)

        kernel_size = weight.size()[2:4]
        kernel_flat_size = int(kernel_size[0] * kernel_size[1])
        in_channel = int(input.size()[1])
        out_channel = int(weight.size()[0])
        print("input.size()", input.size())
        # weight = weight.permute(0, 1, 2, 3)
        print("weight.size()", weight.size())

        # データ数, 1kernelのニューロン数, kernelの数
        col_input = F.unfold(input, kernel_size, dilation, padding, stride)
        # -> torch.Size([50, 243, 14400])
        col_weight = weight.permute(1, 2, 3, 0).view(in_channel * kernel_flat_size, out_channel)
        # -> torch.Size([243, 32])
        col_weight = col_weight.repeat(1, kernel_flat_size).permute(1, 0)
        # -> torch.Size([243, 2592])

        print("unfold_input.size()", col_input.size())
        print("unfold_weight.size()", col_weight.size())
        res = torch.matmul(col_weight, col_input)
        # [50, 2592, 243] @ [50, 243, 14400] = [50, 2592, 14400]
        print("bias.size()", bias.size())
        res = res + bias
        print("conv_out.size()", res.size())
        # -> torch.Size([50, 2592, 14400])
        res = F.fold(res, weight[1], kernel_size, dilation, padding, stride)
        # -> torch.Size([50, 32, 128, 128])
        return res

    # @staticmethod
    # def backward(ctx, *args: Tensor) -> tuple:
    #     pass


# sample
# from torch import nn
#
#
# def run():
#     in_channels = 2
#     out_channels = 5
#     size = 4
#     torch.manual_seed(123)
#     X = torch.rand(10, in_channels, size, size)
#     conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
#     out = conv(X)
#     print('out', out)
#     print('out.size()', out.size())
#     print('')
#
#     Xunfold = F.unfold(X, kernel_size=2, padding=0)
#     # Xunfold = F.unfold(X, kernel_size=3, padding=1)
#     print('X.size()', X.size())
#     print('Xunfold.size()', Xunfold.size())
#
#
# if __name__ == '__main__':
#     run()
