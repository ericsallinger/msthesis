import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle

class WorkloadFrame(Dataset):
    def __init__(self, dir: str, group: str, resample: str, context_length: float):
        if not (0.0 < context_length <= 1.0):
            raise ValueError(f'0.0 < context_length <= 1.0, got {context_length} instead')

        file = {'cog':'/cog_data.pkl',
                'phys':'/phys_data.pkl',
                'tot':'/tot_data.pkl'}[group]
        
        # mapping of signal channel names to numpy columns
        c = {'hr':0, 'hbo':1, 'eda':2, 'hrv':3, 'temp':4}
        self.resample_channel = c[resample]

        ### -------------- DATA CLEANING ------------------- ###

        with open(dir+file, 'rb') as f:
            df = pickle.load(f)

        signal_data = df[list(c.keys())]
        labels = df[['subj', 'label']]

        # trials are removed that are found to have more than 90% nan values in any one channel
        all_nan = signal_data.map(lambda x: self.find_long_nan_sequences(x) > 0.9)
        rows_with_missing_channels = all_nan.apply(lambda x: x.any(), axis=1)

        signal_data = signal_data[~rows_with_missing_channels]
        labels = labels[~rows_with_missing_channels]

        # for each channel, the nan-to-signal-length ratio is computed and it's variance across all channels
        nan_variance = signal_data.apply(lambda y: np.var([y.map(lambda x: np.sum(np.isnan(x)) / len(x))]), axis=1)

        # the median absolute deviation measures dispersion of values around the median. Taking the z-score, the convention is to flag Zmad > 3 as potential outliers
        nan_var_med = nan_variance.median()
        mad = np.median(np.abs(nan_variance - nan_var_med))
        mad_z_scores = np.abs(nan_variance - nan_var_med) / (1.4826 * mad)
        outliers = mad_z_scores > 3

        # remove trials with outlier z-score higher than 3
        signal_data = signal_data[~outliers]
        labels = labels[~outliers]

        # all signals have trailing nan values 
        nan_to_length_ratio = signal_data.map(lambda x: np.sum(np.isnan(x)) / len(x)).apply(lambda x: x.max(), axis=1)
        percent_nan = np.array([nan_to_length_ratio.values for _ in range(5)]).T
        arr_length = signal_data.map(lambda x: len(x))
        missing_vals = percent_nan * arr_length.values

        # empty array is populated with truncated lists
        signal_np_truncated = np.empty((730, 5), dtype=object)

        for i in range(730):
            for j in range(5):
                idx = int(missing_vals[i, j]) 
                seq = signal_data.iloc[i, j]  
                if idx < len(seq):  
                    signal_np_truncated[i, j] = seq[:-idx]
                else:
                    raise ValueError(f'There are {idx} nan values in a sequence of lenght {len(seq)}')

        signal_data_truncated = pd.DataFrame(signal_np_truncated, columns=signal_data.columns)

        ### -------------- SAVE SIGNAL FROM EACH TRIAL ------------------ ###

        # shortest signal timeframe length, context_length cannot exceed this timeframe
        min_timeframe = signal_data_truncated[resample].map(lambda x: len(x)).min()
        self.context_length = int(context_length * min_timeframe - 1)
        
        # save as class attributes
        self.features = signal_data_truncated.values
        self.labels = labels.values

        # all trials are saved as tensors in lists
        self.X = []
        self.Y = []

        self.index_map = [] 
        self.cur_f_idx = None

        for f_idx, signal in enumerate(self.features):

            self.X.append(torch.transpose(torch.stack([self.stretch_arr(self.fill_nan_running_mean(channel), len(signal[self.resample_channel])) for channel in signal]), 0, 1))
            self.Y.append(F.one_hot(torch.tensor(self.labels[f_idx][1], dtype=torch.long), num_classes=4))

            for t_step in range(0, len(signal[self.resample_channel]) - self.context_length, self.context_length // 2):
                self.index_map.append((f_idx, t_step))

    def stretch_arr(self, t, target_len):
        orig_len = len(t)
        orig_idx = np.linspace(0, target_len - 1, orig_len)
        new_idx = np.arange(target_len)

        interpolated = np.interp(new_idx, orig_idx, t)

        return torch.tensor(interpolated, dtype=torch.float32)

    def find_long_nan_sequences(self, arr):
        nan_mask = np.isnan(arr) 

        # exception
        if np.all(nan_mask):
            return 1.0

        
        max_seq = 0
        i = 0
        
        for is_nan in nan_mask:
            if is_nan:
                # increase subsequence counter for each consecutive nan value
                i = i+1
            else:
                # if sequence is broken, save seq length and reset counter
                # this way, nan sequences all the way to the end are not included in the count 
                max_seq = max(i, max_seq)
                i = 0

        return max_seq / len(arr)

    def fill_nan_running_mean(self, arr):
        mask = np.isnan(arr)
        if np.all(mask): 
            return np.zeros_like(arr)
        
        # cumulative sum of array
        cumsum = np.nancumsum(arr)
        # cumulative count of valid elements
        count = np.cumsum(~mask)
        running_mean = np.divide(cumsum, count, where=(count != 0))

        arr[mask] = running_mean[mask]
        return arr

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        f_idx, t_step = self.index_map[idx]

        s, f = t_step, t_step+self.context_length
        x = self.X[f_idx][s:f].unsqueeze(0)
        y = self.Y[f_idx]
        
        return x, y
    

if __name__ == '__main__':

    # directory to .mat files
    dir='C:\\Users\\arthu\\repo\\msthesis\\files'

    #  file group: 'phys', 'cog', or 'tot'
    group='phys'

    # signal channel to resample to: 'temp', 'hrv, 'hr', 'hbo', 'eda'
    resample='temp'

    # size of sliding window relative to shortest signal length; always 50% overlap between windows
    context_length=0.5

    frames = WorkloadFrame(dir=dir, group=group, resample=resample, context_length=context_length)

    # whats happening in the __getitem__ hidden method used by the pytorch dataloader
    # 1st output: the shape of the tensor loaded into memory containing the signal data from one trial
    # 2nd output: the sliding window extracted from that signal data tensor
    print(
        frames.X[0].shape,
        frames.__getitem__(4)[0].shape
    ) 

    # example of resampling an hbo signal to the frequency of the temp signal
    print(
        len(frames.features[0][1]), 'stretched to the following array', frames.stretch_arr(frames.features[0][1], len(frames.features[0][4]))
    )