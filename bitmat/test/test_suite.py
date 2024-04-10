import time
import torch
from bitmat.utils.bitmat import *
from bitmat.utils.packing import *
from bitmat.utils.custom_autotune import *
from bitmat.bitlinear import BitLinear
from ..triton_kernels.bitmat_kernel import bitmat_

def test_kernel_bitlinear():
    class FakeRMSLayerNorm(torch.nn.Module):
        def __init__(self, size):
            super(FakeRMSLayerNorm, self).__init__()
            self.weight = torch.nn.Parameter(torch.ones(size))
            self.size = size

        def forward(self, x):
            return x

    x = torch.randint(-128, 128, (1, 128, 4096), dtype=torch.int8).cuda()
    layer = BitLinear(4096, 16384, bias=False, eps=1e-5 ).cuda().to(torch.float16)
    #we need to block the rms normalization to have a deterministic output
    del layer.norm
    layer.norm = FakeRMSLayerNorm(size=16384)

    torch_layer = torch.nn.Linear(4096, 16384, bias=False).cuda().to(torch.float16)

    torch_layer.weight.data = layer.train().weight.data
    start_time = time.time()
    c = layer(x)
    print("Kenel Time: " + str(time.time() - start_time))
    torch_layer_weight_data, scale_w = terniarize(torch_layer.weight.data)
    torch_layer.weight.data = torch_layer_weight_data.to(torch.float16)
    x, scale_x = quantize_activations(x)
    torch_time = time.time()
    torch_c = torch_layer(x.to(torch.float16)) / scale_w / scale_x
    print("Pytorch Time: " + str(time.time() - torch_time))

    assert (c != torch_c).sum() == 0

test_kernel_bitlinear()

def test_kernel_packing_unpacking():
    w = torch.randint(-1, 2, [4096, 4096], dtype=torch.int8).cuda()
    packed_w = pack_ternary(w, 4)
    unpacked_w = unpack_ternary(packed_w, 4)
    assert (w != unpacked_w).sum() == 0

def test_kernel_bitmat():
    x = torch.randint(-128, 128, (2, 128, 4096), dtype=torch.int8).cuda()
    w = torch.randint(-1, 2, [4096, 4096], dtype=torch.int8).cuda()
    packed_w = pack_ternary(w, 4)

    start_time_16 = time.time()
    c_fp16 = bitmat_(x, packed_w.t().contiguous(),  4, out_dtype=torch.float16) #defoult is float16
    print("Kenel Time c_fp16: " + str(time.time() - start_time_16))

    start_time_32 = time.time()
    c_fp32 = bitmat_(x, packed_w.t().contiguous(),  4, out_dtype=torch.float32)
    print("Kenel Time c_fp32: " + str(time.time() - start_time_32))

    print("c size: ", c_fp16.size())

    torch_time_16 = time.time()
    matmul_fp16 = (x.to(torch.float16) @ w.to(torch.float16).t())
    print("Pytorch Time fp16: " + str(time.time() - torch_time_16))

    torch_time_32 = time.time()
    matmul_fp32 = (x.to(torch.float32) @ w.to(torch.float32).t())
    print("Pytorch Time fp32: " + str(time.time() - torch_time_32))

    print("matmul shape: ", matmul_fp16.shape)

    print("diff in fp16 precision: ", (c_fp16 != matmul_fp16).sum())
    print("diff in fp32 precision: ", (c_fp32 != matmul_fp32).sum())
    print("diff in matmul precision", (matmul_fp16 != matmul_fp32).sum())
    print("diff in kernel precision", (c_fp16 != c_fp32).sum())

# x = torch.randint(-128, 128, (2, 128, 4096), dtype=torch.int8).cuda()
# w = torch.randint(-1, 2, [4096, 4096], dtype=torch.int8).cuda()
# Output:
# Kenel Time c_fp16: 0.7801363468170166
# Kenel Time c_fp32: 0.008198738098144531
# c size:  torch.Size([2, 128, 4096])
# Pytorch Time fp16: 0.0733022689819336
# Pytorch Time fp32: 0.003717660903930664
# matmul shape:  torch.Size([2, 128, 4096])
# diff in fp16 precision:  tensor(230923, device='cuda:0')
# diff in fp32 precision:  tensor(0, device='cuda:0')
# diff in matmul precision tensor(542174, device='cuda:0')
# diff in kernel precision tensor(392754, device='cuda:0')

