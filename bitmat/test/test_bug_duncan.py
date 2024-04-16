
import torch

from transformers import LlamaConfig, LlamaForCausalLM
from bitmat import convert_hf_model

torch.set_default_device('cuda')

seed = 1713219988

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

student_config = LlamaConfig(
  hidden_size=2048,
  intermediate_size=8192,
  num_hidden_layers=24,
  num_attention_heads=32,
  vocab_size=51200,
)

model = LlamaForCausalLM(student_config)

model = convert_hf_model(model)

with torch.no_grad():
  model.eval()

  for i in range(5):
    # torch.cuda.synchronize("cuda") # should fix if it's a sync issue, but does not

    x = model.forward(torch.tensor([[1,2,3]]).to('cuda')).logits # .to("cpu")
    print(i, x) # shows inconsistent results
    # print(i, x[:,:,0]) # no problem
