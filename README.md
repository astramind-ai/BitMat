# BitMat: Improving Ternary Matrix Multiplication with Triton

## 0️⃣1️⃣ Introduction
BitMat is a Python package designed to optimize matrix multiplication operations by utilizing custom kernels written in Triton. Our package leverages the principles outlined in the "1bit-LLM Era" paper, specifically utilizing packed int8 data to enhance computational efficiency and performance in deep learning and numerical computing tasks.

## 🎛 Features
Custom Triton Kernels: Utilize highly optimized kernels for matrix multiplication, tailored for performance and efficiency.

Packed int8 Operations: During inference the model uses packed int8 data to reduce memory usage and improve computational efficiency.

Ease of Integration: BitMat is designed to be easily integrated into existing PyTorch/transformers workflows, providing a seamless user experience.
## 💾 Installation
```bash
pip install bitmat-tl
```
At the moment we only support **Linux** platforms. **Windows** installation is possible but is not tested.
## 🏁 Quick Start

### High-level API (tranformers-compatible)
```python
from transformers import AutoModelForCausalLM
from bitmat import convert_hf_model

# Initialize your model from an available hf model
model= AutoModelForCausalLM.from_pretrained("some-repo/some-model")
# Convert the model to use BitLinear layers
model = convert_hf_model(model)
# Save the converted model
model.save_pretrained('some_local_folder')
```
### Loading the converted 1.58Bit Model
To utilize the converted 1.58Bit model, such as a customized version of Mistral in this exmaple, you will need to import the specific model class from the library. Below is an example demonstrating how to load the Mistral158ForCausalLM model from a local directory:

```python
from bitmat import Mistral158ForCausalLM

# Replace 'path_to_your_model' with the actual path to your model's directory
model = Mistral158ForCausalLM.from_pretrained('path_to_your_model')
```
Once loaded, the model operates in two distinct modes:

- Evaluation Mode: By default, the model employs quantized weights, optimizing performance for inference tasks. Activate this mode using model.eval().

- Training Mode: Switching to this mode, via model.train(), allows the model to leverage full-precision weights, which is essential for training and fine-tuning processes, ensuring accurate gradient calculations and effective model updates.


This API is **fully compatible** with the HuggingFace's Ecosystem 


### Low-level API
```python
import torch
from bitmat import BitLinear

layer = BitLinear(in_features=1024, out_features=512, bias=True, eps=1e-5)
# You can use the layer as a normal torch.nn.Linear layer
```

## 🫱🏼‍🫲🏽 Contributing
We welcome contributions from the community, whether it's adding new features, improving documentation, or reporting bugs. Please refer to our contribution guidelines before making a pull request.

## 📜 License
BitMat is open-sourced under the Apache-2.0 license.

## Citation
If you use BitMat in your research, please cite it using the following Bibtex entry:

```bibtex
@article{bitmat2024,
  title={BitMat: Improving Matrix Multiplication with Custom Triton Kernels},
  author={AstraMind AI},
  journal={https://github.com/astramind-ai/BitMat},
  year={2024}
}
```
## Support
For questions, issues, or support regarding BitMat, please open an issue on our GitHub repository.

## Acknowledgments
Special thanks to the Triton community and the authors of the "1bit-LLM Era" paper for their groundbreaking work and inspiration.

Also thanks to the developer of [BitDelta](https://github.com/FasterDecoding/BitDelta/) and [UnSloth](https://github.com/unslothai/unsloth) since part of the code is based on their work.

