import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import io
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess(raw_data_path, is_train=True, trans=None):
    if is_train:
        raw_data = pd.read_csv(raw_data_path+'train.csv')
    else:
        raw_data = pd.read_csv(raw_data_path+'test.csv')
    images = raw_data.image
    if trans:
        features = [trans(io.read_image(raw_data_path+image))
                     for image in images]
    else:
        features = [io.read_image(raw_data_path+image)
                    for image in images]
    features = torch.stack(features).type(torch.float32)

    if is_train:
        labels = raw_data.label
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        labels = torch.tensor(labels, dtype=torch.long)
        joblib.dump(le, './le.pkl')
        return features, labels
    else:
        return features


class LeaveDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.f = features
        self.label = labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.f[index], self.label[index]

def get_data_loader(raw_data_path, batch_size,num_workers=0, trans=None):
    features, labels = preprocess(raw_data_path,trans=trans)
    num_train = round(0.8 * len(labels))
    train_dataset = LeaveDataset(features[:num_train], labels[:num_train])
    valid_dataset = LeaveDataset(features[num_train:], labels[num_train:])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    )