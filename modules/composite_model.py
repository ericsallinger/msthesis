import torch
import torch.nn as nn

class CompModel(nn.Module):
    def __init__(self, model_dict, hidden_dim, num_classes=4):
        super().__init__()
        self.model_dict = nn.ModuleDict(model_dict)

        self.flatten = nn.Flatten(1, -1)
        latent_dim = sum([self._output_dim(m) for m in self.model_dict.values()])

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def _output_dim(self, model):
        input = torch.randn(5, 1, 700, 5)
        with torch.no_grad():
            out = self.flatten(model(input))
        return out.shape[-1]
    
    def forward(self, x):
        outputs = [self.flatten(model(x)) for model in self.model_dict.values()]
        concat = torch.cat(outputs, dim=-1)
        return self.classifier(concat)



if __name__ == '__main__':
    # imports
    from stat_features.stat_features import StatFeatures
    from wrapper_class import ModelHead
    from frame_dataloader_heavy import WorkloadFrame

    # define heads
    cand_features = ["hr_mean", "hr_std", "hr_skewness", "hr_kurtosis", "hr_slope", "hbo_std", "hbo_slope", "hbo_skewness", "eda_mean", "eda_std", "eda_skewness", "hrv_mean", "hrv_std", "hrv_skewness", "temp_slope"]

    # BATCH DIMENSION != 1 doesn't work :(
    input = torch.randn(89, 1, 700, 5)
    print(len(input.shape))
    
    stats = StatFeatures(cand_features=cand_features)

    model = ModelHead(stats, 'compute')
    output = model(input)
    print('StatFeatures being called by the unified wrapper class', output, output.shape)

    # composite config. keys are used for selecting combinations of models
    config = {'head1':model}

    comp_model = CompModel(config, hidden_dim=36)

    output = comp_model(input)

    print('Composite output', output, output.shape)