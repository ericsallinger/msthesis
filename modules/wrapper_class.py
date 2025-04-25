import torch
import torch.nn as nn

class ModelHead(nn.Module):
    def __init__(self, model, output_fn, output_dim, tune_layers):
        super().__init__()
        self.model = model
        self.output_fn = output_fn
        self.output_dim = output_dim
        #self._freeze(tune_layers)

    def _freeze(self, tune_layers):
        layers = list(self.model.children())
        if tune_layers == -1:
            for p in self.model.parameters():
                p.requires_grad = True
        elif tune_layers == 0:
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            for i, layer in enumerate(layers[:-tune_layers]):
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x):
        if callable(self.output_fn):
            return self.output_fn(self.model, x)
        elif isinstance(self.output_fn, str):
            return getattr(self.model, self.output_fn)(x)
        else:
            raise ValueError("Invalid output_fn: must be string or callable")

    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode)

    def eval(self):
        self.train(False)