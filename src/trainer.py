import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, nn.Module):
        net.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for img, y in data_iter:
            img, y = img.to(device), y.to(device)
            y_hat = net(img)

            y_hat = y_hat.argmax(dim=1)
            correct += (y_hat.reshape(y.shape) == y).sum().item()
            total += y.numel()

    return correct / total

def try_gpu(i=0):
    if torch.cuda.device_count() > i:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def train_epoch(net, train_iter, loss:nn.CrossEntropyLoss, updater:optim.Optimizer, device):
    if isinstance(net, nn.Module):
            net.train()
    l_sum, total = 0, 0
    for img, y in train_iter:
        img, y = img.to(device), y.to(device)
        y_hat = net(img)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        l_sum += l.sum().item()
        total += y.numel()

    return l_sum / total

def train(net, train_iter, valid_iter, num_epochs, lr, weight_decay, device):
    net = net.to(device)

    loss = nn.CrossEntropyLoss(reduction='none')
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

        if (epoch+1) % 10 == 0:
            print(f'train acc: {train_accs[-1]:.4f}, valid acc: {valid_accs[-1]:.4f}')
            torch.save(net.state_dict(), f'/content/drive/MyDrive/data/ep{epoch+1}.pth')

    x = [epoch+1 for epoch in range(num_epochs)]
    plt.plot(x, train_ls, label='train loss')
    plt.plot(x, train_accs, label='train acc')
    plt.plot(x, valid_accs, label='valid acc')
    plt.legend()
    plt.savefig('./loss.png')
    plt.show()


