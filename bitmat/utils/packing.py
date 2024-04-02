# Inspired by the amazing work on DeltaBit https://github.com/FasterDecoding/BitDelta/tree/main

import torch


def pack_ternary(x, n_element_in_one_int=4):
    """
    Pack ternary values into integers.
    x: tensor of shape (*, K, N)
    n_element_in_one_int: int, number of elements in one integer
    return: tensor of shape (*, K, N // n_element_in_one_int)
    """
    assert x.shape[-1] % n_element_in_one_int == 0, "K must be divisible by n_bits"
    assert n_element_in_one_int in [4, 8, 16, 32], "n_element_in_one_int must be 4, 8, 16, 32"
    device = x.device
    x_mapped = x.clone()
    x_mapped[x == -1] = 2

    shift = torch.arange(n_element_in_one_int, device=x.device) * 2

    shape = x.shape[:-1]
    x = x_mapped.view(-1, x.shape[-2], x.shape[-1] // n_element_in_one_int, n_element_in_one_int)

    x = x << shift[None, None, None, :]

    x = x.sum(-1)
    x = x.view(*shape, *x.shape[-1:])

    if n_element_in_one_int == 4:
        dtype = torch.int8
    elif n_element_in_one_int == 8:
        dtype = torch.int16
    elif n_element_in_one_int == 16:
        dtype = torch.int32
    else:
        dtype = torch.int64

    return x.to(dtype).to(device)



def unpack_ternary(x, n_bits=4):
    """
    Unpack ternary values from integers.
    x: tensor of shape (*, K // n_bits, N), where K is the total number of ternary values
    n_bits: int, number of ternary values that each element in x represents
    return: tensor of shape (*, K, N)
    """

    # Create a mask for the shifting
    masks = (3 << (2 * torch.arange(n_bits, device=x.device))).view(1, 1, 1, -1)

    # Use broadcasting for the mask
    x_expanded = x.unsqueeze(-1)
    x_expanded = x_expanded * torch.ones_like(masks)

    # Apply mask and shift values
    unpacked = (x_expanded & masks) >> (2 * torch.arange(n_bits, device=x.device)).view(1, 1, 1, -1)

    # Mappa i valori di nuovo a -1, 0, 1
    unpacked = torch.where(unpacked == 2, torch.tensor(-1, device=x.device), unpacked)

    # Riorganizza le dimensioni per ottenere il formato desiderato (*, K, N)
    return unpacked.view(*x.shape[:-1], -1)

