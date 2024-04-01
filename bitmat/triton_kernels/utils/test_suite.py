import time

from ..bitmat_kernel import *

import torch


x = torch.randint(-128,128,(4096,16384), dtype=torch.int8).cuda()

def test_kernel_packing_unpacking():
    w = torch.randint(-1, 1, [16384, 4096], dtype=torch.int8).cuda()
    packed_w = pack_ternary(w,4)
    unpacked_w = unpack_ternary(packed_w, 4)
    assert (w != unpacked_w).sum() == 0

def test_kernel_matmul():
    w = torch.randint(-1, 1, [16384, 4096], dtype=torch.int8).cuda()
    packed_w = pack_ternary(w,4)

    start_time = time.time()
    c = binary_matmul(x, packed_w.squeeze(),4)
    print("Kenel Time: "+str(time.time()-start_time))
    torch_time = time.time()
    matmul = x.to(torch.float16) @ w.to(torch.float16)
    print("Pytorch Time: "+str(time.time()-torch_time))
    assert (c != matmul).sum() == 0

def test_kernel_batchMatmul():
    x = torch.randint(-128,128,(16,4096,16384), dtype=torch.int8).cuda()
    w = torch.randint(-1, 1, [16, 16384, 4096], dtype=torch.int8).cuda()
    packed_w = pack_ternary(w,4)
    start_time = time.time()
    c = binary_bmm(x, packed_w,4)
    print("Kenel Time: "+str(time.time()-start_time))
    torch_time = time.time()
    matmul = x.to(torch.float16) @ w.to(torch.float16)
    print("Pytorch Time: "+str(time.time()-torch_time))
    assert (c != matmul).sum() == 0
