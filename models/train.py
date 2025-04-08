import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vit_vanila import ViTVanilla
def get_cifar10_dataloaders(batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    test_set = datasets.CIFAR10(root = './data', train = False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle = False)

    return train_loader, test_loader
def train_model(model, loss_fn, data_loader, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_corr, train_n = 0, 0
        model.train()
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_corr += torch.sum(preds == labels).item()
            train_n += labels.size(0)

            train_loss += loss.item()

            print('Epoch: %d | Loss: %.4f | Train Acc: %.2f%%' \
                  (epoch, train_loss / i, train_corr / train_n * 100))
            return model
            

def test_model(model, data_loader):
    model.eval()

    correct = 0
    n = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            _, pred = torch.max(output, 1)

            correct += torch.sum(pred == labels).item()
            n += labels.size(0)
        acc = correct/n
        return acc

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTVanilla(img_size = 32,
                       patch_size = 4,
                       in_channels = 3,
                       emb_dim = 128,
                       depth = 6,
                       num_heads = 4,
                       mlp_dim = 256,
                       num_classes = 10).to(device)
    
    learning_rate = 0.001
    num_epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    train_loader, test_loader = get_cifar10_dataloaders()

    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
    model = train_model(model, loss_fn, train_loader, optimizer, num_epochs, device)
    acc_test = test_model(model, test_loader)

    print('Test Accuracy: %.2f' %(acc_test * 100))


