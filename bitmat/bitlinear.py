import torch

from .utils.bitmat import bitmat
from .utils.packed_parameter import PackedParameter
from .utils.rmsnorm import RMSLayerNorm
from .utils.packing import pack_ternary, unpack_ternary
from .utils.bitmat import terniarize

class BitLinear(torch.nn.Module):
    """
    A linear layer that uses packed terniary matrix multiplication.
    """

    def __init__(self, in_features, out_features, bias=None, eps=1e-5, keep_rms_in_32b=False, dtype=torch.float16):
        super(BitLinear, self).__init__()
        self.eps = eps
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.scale_w = torch.nn.Parameter(torch.Tensor(1))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.norm = RMSLayerNorm(out_features, eps) # added this in favor of models like gemma that needs hp normalization
        self.keep_rms_in_32b = keep_rms_in_32b
        self._post_init()


    """
    def a .to() to keep eps precision
    """


    def _post_init(self):
        #crea un var dei parametri del modello così da poter inizializzare i pesi e i bias
        # Inizializza i pesi utilizzando l'inizializzazione di Kaiming
        params = torch.nn.Parameter(torch.zeros((self.out_features, self.in_features), dtype=self.dtype))
        torch.nn.init.kaiming_normal_(params, mode='fan_out', nonlinearity='relu')
        terniarized_val, self.scale_w.data = terniarize(params)
        del params
        self.register_buffer('weight',pack_ternary(terniarized_val))

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def convert_weights_to_parameters(self):
        # Converti i pesi in torch.nn.Parameter di tipo float16 per il training.
        if self.weight.dtype == torch.int8:
            unpacked_weight = unpack_ternary(self.weight)
            half_weight = (unpacked_weight / self.scale_w).to(self.dtype)
            self.weight = torch.nn.Parameter(half_weight)
            self.scale_w = None# <- this is done so that the bitmat kernel knows we're training

    def convert_weights_to_packed(self):
        # Converti i pesi indietro in PackedParameter di tipo int8 dopo il training.
        if isinstance(self.weight, torch.nn.Parameter):
            terniarized_weight, scale_weight = terniarize(self.weight.data)
            packed_weights = pack_ternary(terniarized_weight)
            self.scale_w = torch.nn.Parameter(scale_weight)
            del self.weight # <- this is done so that torch doesn't trow an error when trying to convert the nn.Parameter to PackedParameter
            self.register_buffer('weight', packed_weights)

    def train(self, mode=True):
        super().train(mode)
        device = next(self.parameters()).device
        if mode:
            self.convert_weights_to_parameters()
        else:
            self.convert_weights_to_packed()
        return self.to(device)

    def forward(self, x):
        if self.training and isinstance(self.weight, PackedParameter):
            # Just to make sure the weights are in the right format even if the user forgot to call train()
            self.convert_weights_to_parameters()
        x_dtype = x.dtype
        x = self.norm(x.to(self.norm.weight.dtype)).to(x_dtype)
        output = bitmat(self.weight.data, x, scale_w=self.scale_w)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output
