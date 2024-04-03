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
        self.dtype = dtype
        self.weight = PackedParameter(torch.zeros(out_features, in_features))
        self.scale_w = torch.Tensor(1)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.norm = RMSLayerNorm(out_features, eps) # added this in favor of models like gemma that needs hp normalization
        self.keep_rms_in_32b = keep_rms_in_32b
        self._post_init()

    def to(self, *args, **kwargs):
        # Calls the parent class' to() method
        super(BitLinear, self).to(*args, **kwargs)

        # Shift weight to the specified device/target
        if isinstance(self.weight, PackedParameter):
            # Assicurati che PackedParameter abbia un metodo to() appropriato
            self.weight.to(*args, **kwargs)

        # Shift scale factor to the specified device/target
        if self.scale_w is not None:
            self.scale_w = self.scale_w.to(*args, **kwargs)

        if self.keep_rms_in_32:
            device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
            self.norm.to(device, torch.float32, non_blocking)

        return self

    def cuda(self, *args, **kwargs):
        return self.to('cuda', *args, **kwargs)

    def _post_init(self):
        #crea un var dei parametri del modello così da poter inizializzare i pesi e i bias
        params = torch.nn.Parameter(torch.zeros((self.weight.data.shape), dtype=self.dtype))
        torch.nn.init.kaiming_normal_(params, mode='fan_out', nonlinearity='relu')
        terniarized_val, self.scale_w = terniarize(params)
        del params
        self.weight = PackedParameter(pack_ternary(terniarized_val))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def convert_weights_to_parameters(self):
        # Converti i pesi in torch.nn.Parameter di tipo float16 per il training.
        if isinstance(self.weight, PackedParameter):
            unpacked_weight = unpack_ternary(self.weight.data)
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
            self.weight = PackedParameter(packed_weights)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Ottieni lo state_dict originale usando il metodo della superclasse
        state_dict = super(BitLinear, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Aggiungi il weight scale se necessario
        if hasattr(self, 'scale_w'):
            key = prefix + 'scale_w'
            if keep_vars:
                state_dict[key] = self.scale_w
            else:
                state_dict[key] = self.scale_w.detach()

        # Gestisci i pesi packed (potrebbero già essere gestiti nel super().state_dict() a seconda dell'implementazione di PackedParameter)
        # Assicurati che il peso sia nel formato desiderato
        if isinstance(self.weight, PackedParameter):
            packed_weight_key = prefix + 'weight'
            state_dict[packed_weight_key] = self.weight.data
            if not keep_vars:
                state_dict[packed_weight_key] = state_dict[packed_weight_key].detach()

        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name, attr in self.__dict__.items():
            if isinstance(attr, PackedParameter):
                key = prefix + name
                if key in state_dict:
                    param = state_dict[key]
                    try:
                        with torch.no_grad():
                            attr.data.copy_(param.data)
                    except Exception as ex:
                        error_msgs.append(f'While copying the packed parameter named "{key}", '
                                          f'an exception occurred: {ex.args}.')
                elif strict:
                    missing_keys.append(key)

        # Ora chiama il comportamento standard di _load_from_state_dict per gestire altri parametri
        super(BitLinear, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                     missing_keys, unexpected_keys, error_msgs)

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
            # Solo per sicurezza, in caso il forward venga chiamato direttamente in training senza chiamare .train()
            self.convert_weights_to_parameters()
        x = self.norm(x.to(self.norm.weight.dtype)).to(self.weight.dtype)
        output = bitmat(self.weight, x, scale_w=self.scale_w)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        return output
