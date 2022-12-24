import torch
import torch.nn.functional as F


class SpectralDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, data, device):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.data = data
        self.device = device

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.from_numpy(self.data[ID].qt_spec_resized)[None].to(self.device)
        y = F.one_hot(torch.tensor(self.labels.loc[ID]).to(torch.int64), num_classes=24)

        return X, y


class SpectralDatasetBase(SpectralDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, data, device):
        super().__init__(list_IDs, labels, data, device)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.from_numpy(self.data[ID].qt_spec_base)[None].to(self.device)
        y = F.one_hot(torch.tensor(self.labels.loc[ID]).to(torch.int64), num_classes=24)

        return X, y


class SpectralDataset30sec(SpectralDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, data, device):
        super().__init__(list_IDs, labels, data, device)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.from_numpy(self.data[ID].qt_spec_30sec)[None].to(self.device)
        y = F.one_hot(torch.tensor(self.labels.loc[ID]).to(torch.int64), num_classes=24)

        return X, y


class SpectralDatasetSegmented(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, data, device):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.data = data
        self.device = device

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.from_numpy(self.data[ID].spec)[None].to(self.device)
        y = F.one_hot(torch.tensor(self.labels.loc[ID]).to(torch.int64), num_classes=24)

        return X, y