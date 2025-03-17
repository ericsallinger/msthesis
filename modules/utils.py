import torch
import os
import psutil  # To check system memory usage

def optimal_num_workers():
    """Dynamically selects the optimal number of workers based on CPU & RAM."""
    cpu_cores = os.cpu_count()  # Total CPU cores
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # Free RAM in GB

    num_workers = max(1, cpu_cores // 2) 

    print(f"Using {num_workers} workers (CPU cores: {cpu_cores}, Available RAM: {available_ram:.2f}GB)")
    return num_workers


def num_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params