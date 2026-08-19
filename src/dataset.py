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
        self.images = self.train_data.iloc[:, 0].to_numpy()
        self.label = self.train_data.iloc[:, 1].to_numpy()
        self.trans = trans
        self.label_trans = label_trans

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, i):
        img_path = os.path.join(self.img_file, self.images[i])
        img = io.read_image(img_path).type(torch.float32)
        if self.trans:
            img = self.trans(img)
        label = self.label[i]
        if self.label_trans:
            label = self.label_trans.transform([label])[0]

        return img, torch.tensor(label, dtype=torch.long)

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

class TestImageDataset(Dataset):
    """用于测试数据的 Dataset，不包含标签"""
    def __init__(self, img_file, test_csv_path, trans=None):
        super().__init__()
        self.img_file = img_file
        self.test_data = pd.read_csv(test_csv_path).iloc[:,0].to_numpy()
        self.trans = trans

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, i):
        img_path = os.path.join(self.img_file, self.test_data[i])
        img = io.read_image(img_path).type(torch.float32)
        if self.trans:
            img = self.trans(img)
        return img

def get_test_data_loader(img_file, test_csv_path, batch_size, num_workers=0, trans=None):
    """为测试数据创建 DataLoader"""
    dataset = TestImageDataset(img_file, test_csv_path, trans=trans)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)