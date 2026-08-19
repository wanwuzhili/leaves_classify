import torch
import yaml

import src.dataset as ds


with open('./config/config.yaml', 'r') as f:
    configs = yaml.safe_load(f)

train_iter, valid_iter = ds.get_data_loader(
    img_file=configs['img_file'],train_csv_path=configs['train_csv_path'], 
    batch_size=configs['batch_size'], num_workers=4, need_label_trans=True
    )

for img, label in train_iter:
    print(f'one batch:{img.shape}, label:{label}')
    break