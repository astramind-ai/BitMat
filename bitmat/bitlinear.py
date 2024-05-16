import torch

from .utils.bitmat import bitmat
from .utils.rmsnorm import RMSLayerNorm
from .utils.packing import pack_ternary, unpack_ternary
from .utils.bitmat import terniarize,bitmat_


import torch
import torch.nn as nn
from torch import Tensor

def weight_quant(weight:Tensor, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s = 1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1)
    return result.type(dtype).to(torch.int8), s


def activation_quant(x, num_bits=8):
    dtype = x.dtype
    x = x.float()
    Qn = -(2 ** (num_bits - 1))
    Qp = 2 ** (num_bits - 1) - 1
    s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * s).round().clamp(Qn, Qp)
    return result.type(dtype)


class BitLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=None,
        eps=1e-5,
        keep_rms_in_32b=False,
        dtype=torch.float16,
        packed=False,
        *args,
        **kwargs,
    ):
        super(BitLinear, self).__init__()
        print("Using Fast Bitmat")
        """
        RMSNorm is placed outside BitLinear
        """
        
        self.weight = torch.nn.Parameter(torch.zeros((out_features, in_features)))
        if packed:
            self.convert_weights_to_packed()
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        num_bits = 8
        self.Qp = 127
        

    def convert_weights_to_parameters(self):
        # Converti i pesi in torch.nn.Parameter di tipo float16 per il training.
        if self.weight.dtype == torch.int8:
            unpacked_weight = unpack_ternary(self.weight)
            half_weight = (unpacked_weight / self.scale_w).to(self.dtype)
            self.weight = torch.nn.Parameter(half_weight)
            self.scale_w = (
                None  # <- this is done so that the bitmat kernel knows we're training
            )

    def convert_weights_to_packed(self):
        print("Packing")
        # Converti i pesi indietro in PackedParameter di tipo int8 dopo il training.
        if not isinstance(self.weight, torch.nn.Parameter):
            return 
        
        terniarized_weight, scale_weight = terniarize(self.weight.data)
        packed_weights = pack_ternary(terniarized_weight)
        
        del self.weight  # <- this is done so that torch doesn't trow an error when trying to convert the nn.Parameter to PackedParameter
        self.register_buffer("weight", packed_weights)
        self.register_buffer("scale_w", scale_weight)
            
    def forward(self, input):
        self.input = input
        quant_input = activation_quant(input)
        
        
        s = 127 / (
            input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5) 
        )
        
        if self.weight.dtype!=torch.int8:
            self.convert_weights_to_packed()
            # quant_weight, scale = terniarize(self.weight)
            # wp = pack_ternary(quant_weight.to(torch.int8))
            # del self.weight
            # self.register_buffer("weight",wp)
            # self.scale_w = scale
            # print("COLD")

        wp = self.weight
        scale = self.scale_w
        out = bitmat_(quant_input.half() / s, wp.t().contiguous()).to(torch.float)

        out = out.to(input.dtype)
        out = out / scale
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out 

    def _post_init(self):
        self.convert_weights_to_packed()


class BitLinearx(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=None,
        eps=1e-5,
        keep_rms_in_32b=False,
        dtype=torch.float16,
        packed=False,
        *args,
        **kwargs,
    ):
        super(BitLinear, self).__init__()
        print("Using Fast Bitmat")
        self.eps = eps
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        if packed:
            self.register_buffer("weight" , torch.zeros((out_features, in_features)))
            self.register_buffer("scale_w",torch.Tensor(1))
        else:
            self.weight = torch.nn.Parameter(torch.zeros((out_features, in_features)))
            
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.keep_rms_in_32b = keep_rms_in_32b
        # self._post_init()

    """
    def a .to() to keep eps precision
    """

    def _post_init(self):
        print("POST init")
        # crea un var dei parametri del modello così da poter inizializzare i pesi e i bias
        # Inizializza i pesi utilizzando l'inizializzazione di Kaiming
        # params = torch.nn.Parameter(
        #     torch.zeros((self.out_features, self.in_features), dtype=self.dtype)
        # )
        # torch.nn.init.kaiming_normal_(params, mode="fan_out", nonlinearity="relu")
        # terniarized_val, self.scale_w.data = terniarize(params)
        # del params
        # self.register_buffer("weight", pack_ternary(terniarized_val))

        # if self.bias is not None:
        #     torch.nn.init.constant_(self.bias, 0)

    def convert_weights_to_parameters(self):
        # Converti i pesi in torch.nn.Parameter di tipo float16 per il training.
        if self.weight.dtype == torch.int8:
            unpacked_weight = unpack_ternary(self.weight)
            half_weight = (unpacked_weight / self.scale_w).to(self.dtype)
            self.weight = torch.nn.Parameter(half_weight)
            self.scale_w = (
                None  # <- this is done so that the bitmat kernel knows we're training
            )

    def convert_weights_to_packed(self):
        # Converti i pesi indietro in PackedParameter di tipo int8 dopo il training.
        if isinstance(self.weight, torch.nn.Parameter):
            terniarized_weight, scale_weight = terniarize(self.weight.data)
            packed_weights = pack_ternary(terniarized_weight)
            self.scale_w = torch.nn.Parameter(scale_weight)
            del self.weight  # <- this is done so that torch doesn't trow an error when trying to convert the nn.Parameter to PackedParameter
            self.register_buffer("weight", packed_weights)

    def train(self, mode=True):
        super().train(mode)
        device = next(self.parameters()).device
        if mode:
            self.convert_weights_to_parameters()
        else:
            self.convert_weights_to_packed()
        return self.to(device)

    def forward(self, input):
        self.input = input
        quant_input = activation_quant(input, self.input_bits)
        
        
        s = self.Qp / (
            input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5) + 2e-6
        )
        
        if self.weight.dtype!=torch.int8:
            quant_weight, scale = weight_quant(self.weight, self.weight_bits)
            wp = pack_ternary(quant_weight.to(torch.int8))
            del self.weight
            self.register_buffer("weight",wp)
            self.scale = scale
            print("cold start")
        else:
            wp = self.weight
            scale = self.scale
        out = bitmat_(quant_input.half() / s, wp.t().contiguous()).to(torch.float)

        out = out.to(input.dtype)
        out = out * scale
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out 
