from typing import Tuple
import os
import torch

from ..triton_kernels.bitmat_kernel import bitmat_
from .packing import pack_ternary

BITMAT_QUANT_8BIT_ACTIVATIONS = not os.getenv("BITMAT_QUANT_8BIT_ACTIVATIONS","True").lower() in ('false', '0', 'f')

def terniarize(weight:torch.Tensor):
    dtype = weight.dtype
    weight = weight.float()
    scale = 1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * scale).round().clamp(-1, 1)
    return result.type(dtype).to(torch.int8), scale.to(dtype)


def quantize_activations(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the activations and returns the scale for each row.
    """
    dtype = x.dtype
    scale = (128 / torch.max(x.abs().max(dim=-1).values, torch.tensor(1e-5))).unsqueeze(-1)
    return torch.clamp((x * scale), -128, 127).to(torch.int8), scale.to(dtype)


class BitMat(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, W, X, scale_w=None,quant_8bit_activations=True):
        """
        During the forward pass, we ternarize the weights, pack them and then quantize the activations.
        We then perform the bit matrix multiplication and return the scaled results.
        ternarization:
        scale_w = 1 / mean(abs(W))                              | STE
        W = clip(round(W * scale_w), -1, 1)                     | STE
        packing:
        packed_w = 4 int8 -> 1 int8                             | STE
        quantization:
        scale_x = 127 / max(abs(X))                             | STE
        X = clip(round(X * scale_x), -127, 128)                 | STE
        bit matrix multiplication:
        Y = X @ w_packed.t()                                    | dot product
        Y = Y / scale_w / scale_x)                              | STE
        """
        X, scale_x = quantize_activations(X)
        if not quant_8bit_activations:
            X = X/scale_x
            
        if scale_w is None:
            dtype = W.dtype
            W, scale_w = terniarize(W)
            #packed_w = pack_ternary(W, 4) -> this is actually not efficent atm
            ctx.save_for_backward(X)
            
            y = X.to(dtype) @ W.to(dtype).t()
            #y = batched_bitmat(X, packed_w) -> this is actually not efficent atm
            out = y / scale_w
        else:
            
            y = bitmat_(X, W.t().contiguous())
            out = y / scale_w
        
        if quant_8bit_activations:
            out = out / scale_x
        
        return out


    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        X = ctx.saved_tensors[0]
        # get dW
        grad_W =  (grad_output.transpose(1,2) @ X).mean(dim=0)
        return grad_W, None, None

def bitmat(W: torch.Tensor, X: torch.Tensor, scale_w,quant_8bit_activations=None) -> torch.Tensor:
    quant_8bit_activations =  quant_8bit_activations if quant_8bit_activations is not None else BITMAT_QUANT_8BIT_ACTIVATIONS
    return BitMat.apply(W, X, scale_w,quant_8bit_activations)