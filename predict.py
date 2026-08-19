import torch
import pandas as pd
import joblib
import yaml
import torchvision.transforms as T

import src.model as m
import src.dataset as ds
import src.trainer as trainer


# load configs
with open('./config/config.yaml', 'r') as f:
    configs = yaml.safe_load(f)

# 设置设备
device = trainer.try_gpu()
print(f'Predicting on {device}')

# load model
net = m.get_model(configs['num_outputs'])
net.load_state_dict(torch.load(configs['model_path'], map_location=device))
net = net.to(device)
net.eval()

# load test data loader (with batching)
trans = T.Compose(
    T.Resize(224),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
)
test_loader = ds.get_test_data_loader(
    img_file=configs['img_file'],
    test_csv_path=configs['test_csv_path'],
    batch_size=configs['batch_size'],
    num_workers=configs['num_workers'],
    trans=trans
)

# load original test data to get image names
test_data = pd.read_csv(configs['test_csv_path'])

# predict with batching
pred = []
with torch.no_grad():
    for img_batch in test_loader:
        img_batch = img_batch.to(device)
        y_hat = net(img_batch)
        y_pred = y_hat.argmax(dim=1).cpu().numpy()
        pred.extend(y_pred)

# convert predictions back to label names
le = joblib.load('./le.pkl')
pred_labels = le.inverse_transform(pred)

# save results
test_data['label'] = pred_labels
test_data.to_csv('submission.csv', index=False)
print(f'Predictions saved to submission.csv')