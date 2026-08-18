import os
import time

import numpy as np
import torch

def save_packed_tensor(filename, tensor):
    """use np.savez_compressed to save compressed tensor and its shape"""
    if tensor.dtype != torch.bool:
        raise TypeError("Input tensor must be of dtype torch.bool")
    shape_array = np.array(tensor.shape)
    packed_data = np.packbits(tensor.numpy())
    tmp = f'{filename}.tmp.{os.getpid()}.{time.time_ns()}'
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, shape=shape_array, data=packed_data)
    os.replace(tmp, filename)

def load_packed_tensor(filename):
    """read .npz file and decompress tensor"""
    with np.load(filename) as loader:
        shape = loader['shape']
        packed_data = loader['data']
    numel = np.prod(shape)
    unpacked_data = np.unpackbits(packed_data, count=numel)
    restored_tensor = torch.from_numpy(unpacked_data.reshape(shape))
    return restored_tensor
