import torch
import time
import matplotlib.pyplot as plt

from ..bitmat_kernel import *


sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
kernel_times = []
torch_times = []

for size in sizes:
    w = torch.randint(-1, 1, [size, size], dtype=torch.int8).cuda()
    x = torch.randint(-128, 128, [size, size], dtype=torch.int8).cuda()

    # Packing
    packed_w = pack_ternary(w, 4)

    # Kernel personalizzato
    start_time = time.time()
    c = binary_matmul(x, packed_w.squeeze(), 4)
    kernel_times.append(time.time() - start_time)

    # PyTorch matmul
    torch_time = time.time()
    matmul = x.to(torch.float16) @ w.to(torch.float16)
    torch_times.append(time.time() - torch_time)

kernel_flops_per_second = []
torch_flops_per_second = []

for size in sizes:
    w = torch.randint(-1, 1, [size, size], dtype=torch.int8).cuda()
    x = torch.randint(-128, 128, [size, size], dtype=torch.int8).cuda()

    # Packing
    packed_w = pack_ternary(w, 4)

    # Matmul FLOP
    flops = size * size * size + (size - 1) * size * size

    # Custom Kernel
    torch.cuda.synchronize()
    start_time = time.time()
    c = binary_matmul(x, packed_w.squeeze(), 4)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    kernel_flops_per_second.append(flops / elapsed_time / 1e12)

    # PyTorch matmul
    torch.cuda.synchronize()
    torch_time = time.time()
    matmul = x.to(torch.float16) @ w.to(torch.float16)
    torch.cuda.synchronize()
    elapsed_time = time.time() - torch_time
    torch_flops_per_second.append(flops / elapsed_time / 1e12)



# Graphs
plt.figure(figsize=(10, 6))
plt.plot(sizes, kernel_times, label='Custom Kernel Matmul')
plt.plot(sizes, torch_times, label='PyTorch Matmul')
plt.xlabel('Matrix Size')
plt.ylabel('Execution time (s)')
plt.title('Difference between Custom Kernel and PyTorch Matmul')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sizes, kernel_flops_per_second, label='Custom Kernel', marker='o')
plt.plot(sizes, torch_flops_per_second, label='PyTorch Matmul', marker='x')
plt.xlabel('Matrix Size')
plt.ylabel('TFLOPS')
plt.title('Differences in TFLOPS between Custom Kernel and PyTorch Matmul')
plt.legend()
plt.grid(True)
plt.show()
