import torch
import yaml

import src.dataset as ds


with open('./config/config.yaml', 'r') as f:
    configs = yaml.safe_load(f)

train_iter, valid_iter = ds.get_data_loader(
    raw_data_path=configs['raw_data_path'], batch_size=configs['batch_size'],
    )

for img, label in train_iter:
    print(f'one batch:{img.shape}, label:{label}')
    break