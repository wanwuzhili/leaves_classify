import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import io
from sklearn.preprocessing import LabelEncoder
import joblib


class ImageDataset(Dataset):
    def __init__(self, img_file, train_csv_path, trans=None, label_trans:LabelEncoder=None):
        super().__init__()
        self.img_file = img_file
        self.train_data = pd.read_csv(train_csv_path)
        self.trans = trans
        self.label_trans = label_trans

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, i):
        img_path = os.path.join(self.img_file, self.train_data.iloc[i, 0])
        img = io.read_image(img_path)
        if self.trans:
            img = self.trans(img)
        label = self.train_data.iloc[i, 1]
        if self.label_trans:
            label = self.label_trans.transform([label])[0]

        return img.type(torch.float32), torch.tensor(label, dtype=torch.long)

def label_encoder(train_csv_path):
    labels = pd.read_csv(train_csv_path).iloc[:, 1]
    le = LabelEncoder()
    le = le.fit(labels)
    joblib.dump(le, './le.pkl')
    return le

def get_data_loader(img_file, train_csv_path, batch_size,
                    num_workers=0, trans=None, need_label_trans=True):
    if need_label_trans:
        le = label_encoder(train_csv_path)
    else:
        le = None
    dataset = ImageDataset(img_file, train_csv_path, trans=trans, label_trans=le)
    num = dataset.__len__()
    num_train = round(0.8 * num)
    num_valid = num - num_train
    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )