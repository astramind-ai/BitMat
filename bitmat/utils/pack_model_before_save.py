from bitmat.bitlinear import BitLinear
from bitmat.utils.bitmat import terniarize
from bitmat.utils.packing import pack_ternary


def pack_ternary_model(model, n_element_in_one_int=4):
    """
    Pack every BitLinear layer in the model.
    """
    for name, module in model.named_children():
        if isinstance(module, BitLinear):
            module.convert_weights_to_packed()

        else:
            pack_ternary_model(module, n_element_in_one_int=n_element_in_one_int)
    return model
