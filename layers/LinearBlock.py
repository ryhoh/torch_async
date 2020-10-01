from typing import Any

from torch import nn


class HorizontalLinearBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(HorizontalLinearBlock, self).__init__()
        node_n = int(out_size ** 0.5)
        assert node_n ** 2 == out_size  # temporary
        self.layers = [nn.modules.Linear(in_size, out_size, bias=True) for _ in range(node_n)]

    def forward(self, x):


    def _forward_unimplemented(self, *input: Any) -> None:
        pass
