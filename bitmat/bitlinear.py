import torch

from .utils.bitmat import bitmat
from .utils.rmsnorm import RMSLayerNorm

class BitLinear(torch.nn.Module):
    """
    A linear layer that uses packed terniary matrix multiplication.
    """
    def __init__(self, in_features, out_features, eps, bias=None):
        super(BitLinear, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.norm = RMSLayerNorm(out_features, 1e-5)
        self._post_init()
    def _post_init(self):
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = self.norm(x)
        output = bitmat(self.weight, x)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output


