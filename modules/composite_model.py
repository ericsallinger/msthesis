import torch
import torch.nn as nn

class CompModel(nn.Module):
    def __init__(self, model_list, hidden_dim, latent_dim, num_classes=4):
        super().__init__()
        self.model_list = nn.ModuleList(model_list)

        tot_output_dim = sum([m.output_dim for m in self.model_list])

        # classification head
        self.flatten = nn.Flatten(1, -1)
        self.classifier = nn.Sequential(
            nn.Linear(tot_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # TODO: projections to normalize dim
        # self.projections = nn.ModuleDict({
        #     i: nn.Linear(self.model_list[i].ouput_dim, latent_dim * int(self.model_list[i].output_dim // tot_output_dim))
        #     for i in range(len(self.model_list))
        # })

        # TODO: learned weightings
        # attention_weights = nn.Parameter(torch.ones(len(self.model_list)))
        # features = [proj(model(x)) for model, proj in zip(self.model_list, self.projections)]
        # weighted_sum = sum(w * f for w, f in zip(attention_weights, features))

        # TODO: nn.Dropout()
    
    def train(self, mode=True):
        super().train(mode)
        for m in self.model_list:
            m.train(mode)

    def eval(self):
        super().train(False)
        for m in self.model_list:
            m.train(False)
        
    def encode(self, x):
        outputs = [self.flatten(model(x)) for model in self.model_list]
        concat = torch.cat(outputs, dim=-1)
        return concat
    
    def classify(self, concat):
        return self.classifier(concat)

    def forward(self, x):
        concat = self.encode(x)
        return self.classify(concat)
    
if __name__ == '__main__':
    from wrapper_class import ModelHead
    from torch.utils.data import DataLoader
    from stat_features.stat_features import StatFeatures

    stats_cfg = {'cand_features':["hr_mean", "hr_std", "hr_skewness", "hr_kurtosis", "hr_slope", "hbo_std", "hbo_slope", "hbo_skewness", "eda_mean", "eda_std", "eda_skewness", "hrv_mean", "hrv_std", "hrv_skewness", "temp_slope"]}
    stats = {'model':StatFeatures(**stats_cfg), 'output_fn':'compute', 'output_dim':len(stats_cfg['cand_features']), 'tune_layers':0}

    cl = CompModel([ModelHead(**stats)], 21, None)

    sample_x, sample_y = torch.randn(size=(5, 1, 129, 5)), torch.nn.functional.one_hot(torch.randint(0, 3, (5,)), num_classes=4)

    out = cl(sample_x)
    loss = torch.nn.functional.cross_entropy(out, sample_y.float())
    print(out.shape, loss)
    loss.backward()
