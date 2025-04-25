# computes the following stats: mean, std, slope, skewness, kurtosis
# candidate features: hr_mean, hr_std, hr_skewness, hr_kurtosis, hr_slope, hbo_std, hbo_slope, hbo_skewness, eda_mean, eda_std, eda_skewness, hrv_mean, hrv_std, hrv_skewness, temp_slope
import torch
import torch.nn as nn

class StatFeatures(nn.Module):
    def __init__(self, cand_features=None):
        super().__init__()
        self.cand_features = cand_features

    def compute(self, signal):
        """
        Takes signal of shape (N, C, T, S), (C, T, S), or (T, S) where T is the context length and S is the number of signals (5).
        Computes cand_features:List or (mean, std, slope, skewness, kurtosis) statistics over T
        """

        channel = {'hr':0, 'hbo':1, 'eda':2, 'hrv':3, 'temp':4}
        stat = {'mean':0, 'std':1, 'slope':2, 'skewness':3, 'kurtosis':4}

        if len(signal.shape) == 2:
            dim = 0
        elif len(signal.shape) == 3:
            dim = 1
        elif len(signal.shape) == 4:
            dim = 2
        else:
            raise ValueError('Signal has unrecognized shape')

        mean = torch.mean(signal, dim=dim, keepdim=True)
        std = torch.std(signal, dim=dim, keepdim=True)
        slope = torch.mean(torch.gradient(signal, dim=dim, spacing=1)[0], dim=dim)

        diffs = signal - mean
        zscores = (diffs / (std + 1e-6))

        mean = mean.squeeze(dim)
        std = std.squeeze(dim)

        skewness = torch.mean(torch.pow(zscores, 3.0), dim=dim)
        kurtosis = torch.mean(torch.pow(zscores, 4.0), dim=dim) - 3.0 

        stats = torch.stack([mean, std, slope, skewness, kurtosis], dim=-1)

        if self.cand_features:
            if len(signal.shape) == 2:
                stat_features = torch.stack([stats[channel[s.split('_')[0]], stat[s.split('_')[1]]] for s in self.cand_features], dim=-1)
            elif len(signal.shape) == 3:
                stat_features = torch.stack([stats[:, channel[s.split('_')[0]], stat[s.split('_')[1]]] for s in self.cand_features], dim=-1)
            elif len(signal.shape) == 4:
                stat_features = torch.stack([stats[:, :, channel[s.split('_')[0]], stat[s.split('_')[1]]] for s in self.cand_features], dim=-1)
        else:
            stat_features = stats

        return stat_features
    
