import torch
import numpy as np

def model_size_in_bytes(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()  # Number of elements in the parameter
    # Assuming 32-bit (4 bytes) precision for each parameter
    total_size_bytes = total_params * 4
    return total_size_bytes