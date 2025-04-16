import torch
import torch.nn as nn

class ModelHead(nn.Module):
    def __init__(self, model, output_fn):
        super().__init__()
        self.model = model
        self.output_fn = output_fn

    def forward(self, x):
        if callable(self.output_fn):
            return self.output_fn(self.model, x)
        elif isinstance(self.output_fn, str):
            return getattr(self.model, self.output_fn)(x)
        else:
            raise ValueError("Invalid output_fn: must be string or callable")
