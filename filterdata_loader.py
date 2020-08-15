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
        f = self.feature[idx, :]
        l = self.label[idx, :]

        # convert to tensor
        f = torch.from_numpy(f.copy()).float()
        l = torch.from_numpy(l.copy()).float()
        l = l.unsqueeze(1)

        return f, l


if __name__ == "__main__":
    train_data = FilterDataset(feature_path=train_feature, label_path=train_label)

    # show a batch
    batch_size = 5

    for i in range(batch_size):
        feature, label = train_data[i]
        print(i, feature.size(), label.size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch[0].size(), batch[1].size())
