import torch
from ..triton_kernels.rmsnorm_kernel import fast_rms_layernorm


class RMSLayerNorm(torch.nn.Module):
    def __init__(self, shape, eps, dtype=torch.float16):
        super(RMSLayerNorm, self).__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.randn(shape))

    def forward(self, x):
        return fast_rms_layernorm(self.weight, x, self.eps).to(x.dtype)
