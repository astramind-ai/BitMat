import torch
from torch import nn

from ..bitlinear import BitLinear

from transformers import AutoModel, GemmaForCausalLM, MistralForCausalLM, LlamaForCausalLM

# Importing custom hijack classes for specific models
from .model_hijacks.gemma_hijack import HijackedGemmaForCausalLM
from .model_hijacks.mistral_hijack import HijackedMistralForCausalLM
from .model_hijacks.llama_hijack import HijackedLlamaForCausalLM

def convert_hf_model(model: AutoModel) -> AutoModel:
    """
    Convert a Hugging Face model to use BitLinear layers instead of Linear layers.
    """
    # Check the type of the model and initialize the corresponding hijacked model
    if isinstance(model, GemmaForCausalLM):
        hijacked_model = HijackedGemmaForCausalLM(model.config)
    elif isinstance(model, MistralForCausalLM):
        hijacked_model = HijackedMistralForCausalLM(model.config)
    elif isinstance(model, LlamaForCausalLM):
        hijacked_model = HijackedLlamaForCausalLM(model.config)
    else:
        raise RuntimeError("Unsupported model type.")

    # Load the original model's state dict into the hijacked model
    hijacked_model.load_state_dict(model.state_dict(), strict=False)

    # Apply the BitLinear conversion to the hijacked model
    apply_bitlinear_to_hf_model(hijacked_model)
    print("Model converted successfully")
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
            bit_linear = BitLinear(module.in_features, module.out_features, rms_layers.get(f"{name}_rms", {}).get('eps', torch.tensor(1e-5)), module.bias is not None)
            setattr(model, name, bit_linear)
        else:
            # Recursively apply to child modules
            apply_bitlinear_to_hf_model(module, full_name)

    return model
