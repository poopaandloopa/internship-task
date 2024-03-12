import torch
from torch.utils.data import Dataset


class EthanolLevelDataset(Dataset):
    def __init__(self, dataframe):
        dataframe.rename(columns={0: "Class"}, inplace=True)

        self.target = torch.LongTensor(dataframe["Class"].values) - 1
        self.data = torch.Tensor(dataframe.drop("Class", axis=1).values)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
