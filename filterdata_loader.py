import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


root_dir = ""
train_feature = os.path.join(root_dir, 'x_train.npy')
train_label = os.path.join(root_dir, 'y_train.npy')
val_feature = os.path.join(root_dir, "x_test.npy")
val_label = os.path.join(root_dir, 'y_test.npy')

"""
batch_size should equal to n: 100
"""

class FilterDataset(Dataset):

    def __init__(self, feature_path, label_path):
        self.feature = np.load(feature_path)
        self.label = np.load(label_path)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        v = self.feature[idx, :, 0]
        a = self.feature[idx, :, 0]
        label = self.label[idx, :]

        # convert to tensor
        v = torch.from_numpy(v.copy()).float()
        a = torch.from_numpy(a.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        sample = {'v': v, 'a': a, 'l': label}

        return sample



if __name__ == "__main__":
    train_data = FilterDataset(feature_path=train_feature, label_path=train_label)

    # show a batch
    batch_size = 5

    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['v'].size(), sample['a'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['v'].size(), batch['a'].size())

