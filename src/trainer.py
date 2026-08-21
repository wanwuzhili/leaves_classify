import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, nn.Module):
        net.eval()
    accs = []
    with torch.no_grad():
        for img, y in data_iter:
            img, y = img.to(device), y.to(device)
            y_hat = net(img)

            y_hat = y_hat.argmax(dim=1)
            cmp = y_hat.reshape(y.shape).type(y.dtype) == y
            acc = torch.sum(cmp) / len(cmp)
            accs.append(acc.item())

    return sum(accs) / len(accs)

def try_gpu(i=0):
    if torch.cuda.device_count() > i:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def train_epoch(net, train_iter, loss:nn.CrossEntropyLoss, updater:optim.Optimizer, device):
    if isinstance(net, nn.Module):
            net.train()
    ls = []
    for img, y in train_iter:
        img, y = img.to(device), y.to(device)
        y_hat = net(img)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        ls.append(l.mean().detach().item())

    return sum(ls) / len(ls)

def train(net, train_iter, valid_iter, num_epochs, lr, weight_decay, device):
    net = net.to(device)

    loss = nn.CrossEntropyLoss()
    updater = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_ls = []
    train_accs, valid_accs = [], []
    print(f'train on {device}')
    for epoch in range(num_epochs):
        l_epoch = train_epoch(net, train_iter, loss, updater, device)
        print(f'epoch: {epoch+1}, loss: {l_epoch:.4f}')
        train_ls.append(l_epoch)
        train_accs.append(evaluate_accuracy(net, train_iter, device))
        valid_accs.append(evaluate_accuracy(net, valid_iter, device))

    print(f'train acc: {train_accs[-1]:.4f}, valid acc: {valid_accs[-1]:.4f}')
    x = [epoch+1 for epoch in range(num_epochs)]
    plt.plot(x, train_ls, label='train loss')
    plt.plot(x, train_accs, label='train acc')
    plt.plot(x, valid_accs, label='valid acc')
    plt.legend()
    plt.savefig('./loss.png')
    plt.show()


