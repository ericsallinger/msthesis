import os
import psutil  # To check system memory usage
import json
import numpy as np
import skopt # skopt.space.Space, skopt.space.Integer etc...

# TODO: add class decorator for showing/hiding model print statements

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

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to convert NumPy types to JSON-serializable types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):  # Convert arrays to lists
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):  # Convert NumPy numbers to Python types
            return obj.item()
        if isinstance(obj, bytes):  # Convert bytes to string
            return obj.decode()
        # if isinstance(obj, skopt.space.Integer) or isinstance(obj, skopt.space.Real):
        #     return {d:(r.low, r.high, r.name, str(r.dtype)) for d, r in enumerate(obj)}
        # if isinstance(obj, skopt.space.Categorical):
        #     return {d:(r.categories, r.name) for d, r in enumerate(obj)}
        return super().default(obj)

def make_serializable_dict(result):
    """Filter out non-serializable objects from OptimizeResult."""
    serializable_dict = {}
    
    for key, value in result.items():
        try:
            # Attempt to serialize the value using JSON
            json.dumps(value, cls=NumpyEncoder)
            serializable_dict[key] = value  # Keep it if serialization works
        except (TypeError, OverflowError):
            pass  # Skip non-serializable entries
    
    return serializable_dict

def save_optimize_result(result, filepath, filename):
    """Remove non-serializable objects and save to JSON."""
    serializable_result = make_serializable_dict(result)
    
    with open(filepath + filename, "w") as f:
        json.dump(serializable_result, f, indent=4, cls=NumpyEncoder)