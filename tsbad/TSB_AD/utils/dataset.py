import torch
import torch.utils.data
import numpy as np
epsilon = 1e-8

class ReconstructDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max(0, (self.data.shape[0] - window_size) // stride + 1)
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)

        if self.univariate:
            data = data.squeeze()
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size] for i in range(self.sample_num)])
            X = X.unsqueeze(-1)
        else:
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size, :] for i in range(self.sample_num)])

        return X, X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

class ForecastDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, pred_len, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.pred_len = pred_len
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size - pred_len) // stride + 1, 0)

        # Generate samples efficiently
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        """ Normalize data using mean and standard deviation. """
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        """ Generate windowed samples efficiently using vectorized slicing. """
        data = torch.tensor(self.data, dtype=torch.float32)

        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        X = torch.stack([data[i : i + self.window_size] for i in indices])
        Y = torch.stack([data[i + self.window_size : i + self.window_size + self.pred_len] for i in indices])

        return X, Y  # Inputs & targets

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

# class ForecastDataset(torch.utils.data.Dataset):
#     def __init__(self, data, window_size, pred_len, normalize=True):
#         super().__init__()
#         self.normalize = normalize

#         if self.normalize:
#             data_mean = np.mean(data, axis=0)
#             data_std = np.std(data, axis=0)
#             data_std = np.where(data_std == 0, epsilon, data_std)
#             self.data = (data - data_mean) / data_std
#         else:
#             self.data = data

#         self.window_size = window_size
        
#         if data.shape[1] == 1:
#             data = data.squeeze()
#             self.len, = data.shape
#             self.sample_num = max(self.len - self.window_size - pred_len + 1, 0)
#             X = torch.zeros((self.sample_num, self.window_size))
#             Y = torch.zeros((self.sample_num, pred_len))
            
#             for i in range(self.sample_num):
#                 X[i, :] = torch.from_numpy(data[i : i + self.window_size])
#                 Y[i, :] = torch.from_numpy(np.array(
#                     data[i + self.window_size: i + self.window_size + pred_len]
#                 ))
            
#             self.samples, self.targets = torch.unsqueeze(X, -1), torch.unsqueeze(Y, -1)
    
#         else:
#             self.len = self.data.shape[0]
#             self.sample_num = max(self.len - self.window_size - pred_len + 1, 0)

#             X = torch.zeros((self.sample_num, self.window_size, self.data.shape[1]))
#             Y = torch.zeros((self.sample_num, pred_len, self.data.shape[1]))

#             for i in range(self.sample_num):
#                 X[i, :] = torch.from_numpy(data[i : i + self.window_size, :])
#                 Y[i, :] = torch.from_numpy(data[i + self.window_size: i + self.window_size + pred_len, :])
            
#             self.samples, self.targets = X, Y

#     def __len__(self):
#         return self.sample_num

#     def __getitem__(self, index):
#         return self.samples[index, :, :], self.targets[index, :, :]

class TSDataset(torch.utils.data.Dataset):

    def __init__(self, X, y=None, mean=None, std=None):
        super(TSDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
            # assert_almost_equal (0, sample.mean(), decimal=1)

        return torch.from_numpy(sample), idx


class ReconstructDataset_Moment(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size) // stride + 1, 0)

        self.samples = self._generate_samples()
        self.input_mask = np.ones(self.window_size, dtype=np.float32)  # Fixed input mask

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        if self.univariate:
            X = torch.stack([data[i : i + self.window_size] for i in indices])
        else:
            X = torch.stack([data[i : i + self.window_size, :] for i in indices])

        return X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.input_mask



class TSPulseFinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        window_size,
        aggr_window_size=None,
        label=None,
        stride=1,
        normalize=True,
        return_dict=False,
        channel_last=True,
    ):
        # label is only used for plotting
        super().__init__()
        self.window_size = window_size

        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        if self.data.shape[0] < window_size:
            pad_len = window_size - self.data.shape[0]
            pad = np.ones((pad_len, self.data.shape[1])) * self.data[:1]
            self.data = np.concatenate((pad, self.data), axis=0)

        self.label = label
        self.return_dict = return_dict
        self.channel_last = channel_last

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max(((self.data.shape[0] - window_size) // stride) + 1, 0)

        if self.label is not None:
            self.samples, self.gen_labels = self._generate_samples()
        else:
            self.samples = self._generate_samples()
        self.input_mask = np.ones((self.window_size, data.shape[1]), dtype=np.float32)  # Fixed input mask
        if aggr_window_size is not None:
            self.input_mask[:aggr_window_size, :] = 0

        if not self.channel_last:
            self.samples = self.samples.permute(0, 2, 1)  # batch, channel, window_size
            self.input_mask = self.input_mask[:, 0]  # window_size

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        if self.univariate:
            X = torch.stack([data[i : i + self.window_size] for i in indices])
            print(X.shape)
        else:
            X = torch.stack([data[i : i + self.window_size] for i in indices])
            print(X.shape)

        if self.label is not None:
            self.label = torch.tensor(self.label)
            Y = torch.stack([self.label[i : i + self.window_size] for i in indices])
            return X, Y

        return X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        if self.label is not None:
            if self.return_dict:
                return {
                    "past_values": self.samples[index],
                    "anomaly_labels": self.gen_labels[index],
                    "past_observed_mask": self.input_mask,
                }
            else:
                return self.samples[index], self.input_mask, self.gen_labels[index]
        else:
            if self.return_dict:
                return {
                    "past_values": self.samples[index],
                    "past_observed_mask": self.input_mask,
                }
            else:
                return self.samples[index], self.input_mask
    