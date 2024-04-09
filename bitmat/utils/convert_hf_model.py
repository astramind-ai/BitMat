import torch
from torch import nn
from tqdm import tqdm
from ..bitlinear import BitLinear

from transformers import AutoModel, GemmaConfig, MistralConfig, LlamaConfig

# Importing custom hijack classes for specific models
from .model_hijacks.gemma_1_58b import Gemma158ForCausalLM
from .model_hijacks.mistral_1_58b import Mistral158ForCausalLM
from .model_hijacks.llama_1_58b import Llama158ForCausalLM


def convert_hf_model(model: AutoModel) -> AutoModel:
    """
    Convert a Hugging Face model to use BitLinear layers instead of Linear layers.
    """
    model_config = model.config
    del model

    # initialize a tqdm progress bar
    pbar = tqdm(total=1, desc="Converting model to 1.58Bit")
    # Check the type of the model and initialize the corresponding hijacked model
    if isinstance(model_config, GemmaConfig):
        hijacked_model = Gemma158ForCausalLM(model_config)
    elif isinstance(model_config, MistralConfig):
        hijacked_model = Mistral158ForCausalLM(model_config)
    elif isinstance(model_config, LlamaConfig):
        hijacked_model = Llama158ForCausalLM(model_config)
    else:
        raise RuntimeError("Unsupported model type. Please open an issue on GitHub citing the model you are using")

    pbar.update(1)
    return hijacked_model


def apply_bitlinear_to_hf_model(model: AutoModel, parent_name='') -> AutoModel:
    """
    Recursively replace Linear layers with BitLinear layers in the model,
    except for layers within 'lm_head'.
    """
    # Store any RMS layer configurations encountered
    rms_layers = {}

    # Remove RMS layers and store their configurations
    for name, module in list(model.named_children()):
        if 'RMS' in type(module).__name__:
            if hasattr(module, 'eps'):
                rms_eps = module.eps
            elif hasattr(module, 'epsilon'):
                rms_eps = module.epsilon
            elif hasattr(module, 'variance_epsilon'):
                rms_eps = module.variance_epsilon
            else:
                raise RuntimeError(
                    "Model type not mappable, please open an issue on GitHub citing the model you are using")

            # save weights and eps
            rms_layers[name] = {'eps': rms_eps}

            delattr(model, name)
    # Replace Linear layers with BitLinear layers
    for name, module in model.named_children():
        full_name = f'{parent_name}.{name}' if parent_name else name

        if 'lm_head' not in full_name and isinstance(module, nn.Linear):
            # Replace Linear layer with BitLinear
            bit_linear = BitLinear(module.in_features, module.out_features,
                                   rms_layers.get(f"{name}_rms", {}).get('eps', torch.tensor(1e-5)),
                                   module.bias is not None)
            setattr(model, name, bit_linear)
        else:
            # Recursively apply to child modules
            apply_bitlinear_to_hf_model(module, full_name)

    return model
