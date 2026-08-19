import yaml
import torch
import torchvision.transforms as T

import src.dataset as ds
import src.model as m
import src.trainer as trainer


# load configs
with open('./config/config.yaml', 'r') as f:
    configs = yaml.safe_load(f)

# load data
trans = T.Compose(
    T.Resize(224),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
)
train_iter, valid_iter = ds.get_data_loader(
    img_file=configs['img_file'], train_csv_path=configs['train_csv_path'],
    batch_size=configs['batch_size'], num_workers=configs['num_workers'], trans=trans
)

# load model
net = m.get_model(num_outputs=configs['num_outputs'])

# train
trainer.train(net, train_iter, valid_iter, num_epochs=configs['num_epochs'],
              lr=configs['lr'], weight_decay=configs['weight_decay'], device=trainer.try_gpu())

# save model
torch.save(net.state_dict(), configs['model_path'])