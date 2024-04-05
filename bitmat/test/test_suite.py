import time

from bitmat.utils.bitmat import *
from bitmat.utils.packing import *
from bitmat.bitlinear import BitLinear
import torch




def test_kernel_bitlinear():
    class FakeRMSLayerNorm(torch.nn.Module):
        def __init__(self, size):
            super(FakeRMSLayerNorm, self).__init__()
            self.weight = torch.nn.Parameter(torch.ones(size))
            self.size = size

        def forward(self, x):
            return x

    x = torch.randint(-128, 128, (15, 128, 4096), dtype=torch.int8).cuda()
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


def test_kernel_packing_unpacking():
    w = torch.randint(-1, 2, [4096, 4096], dtype=torch.int8).cuda()
    packed_w = pack_ternary(w, 4)
    unpacked_w = unpack_ternary(packed_w, 4)
    assert (w != unpacked_w).sum() == 0


def test_kernel_matmul():
    from bitmat.triton_kernels.bitmat_kernel import bitmat
    x = torch.randint(-128, 128, (128, 4096), dtype=torch.int8).cuda()
    w = torch.randint(-1, 2, [4096, 4096], dtype=torch.int8).cuda()
    packed_w = pack_ternary(w, 4)

    start_time = time.time()
    c = bitmat(x, packed_w.t().contiguous(),  4)
    print("Kenel Time: " + str(time.time() - start_time))
    torch_time = time.time()
    matmul = x.to(torch.float16) @ w.to(torch.float16).t()
    print("Pytorch Time: " + str(time.time() - torch_time))
    assert (c != matmul).sum() == 0


def test_kernel_batchMatmul():
    x = torch.randint(-128, 128, (1, 128, 4096), dtype=torch.int8).cuda() #TODO: batch size = 1 seems problermatic. need further investigation
    w = torch.randint(-1, 2, [5000*6, 4096], dtype=torch.int8).cuda()

    packed_w = pack_ternary(w, 4)
    start_time = time.time()
    c = batched_bitmat(x, packed_w, 4)
    print("Kenel Time: " + str(time.time() - start_time))
    torch_time = time.time()
    matmul =x.to(torch.float16) @  w.to(torch.float16).t()
    print("Pytorch Time: " + str(time.time() - torch_time))
    assert (c != matmul).sum() == 0

