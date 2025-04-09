import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
#from vit_vanilla import ViTVanilla


def train_model(model, loss_fn, train_loader, test_loader, optimizer, num_epochs, device, train_acc, test_acc,
                epoch_itr):
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_corr, train_n = 0, 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            _, pred = torch.max(outputs, 1)
            train_corr += torch.sum(pred == labels).item()
            train_n += labels.size(0)

            train_loss += loss.item()

        print('Epoch: %d | Loss: %.4f | Train Acc: %.2f%%' % (
            epoch + 1, train_loss, (train_corr / train_n) * 100))
        acc_test = test_model(model, test_loader)
        print('Test Accuracy: %.2f' % (acc_test * 100))
        train_acc.append(train_corr / train_n)
        test_acc.append(acc_test)
        epoch_itr.append(epoch + 1)
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
        acc = correct / n
        return acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = ViTVanilla(img_size=32,
    #                    patch_size=4,
    #                    in_channels=3,
    #                    emb_dim=128,
    #                    depth=6,
    #                    num_heads=4,
    #                    mlp_dim=512,
    #                    num_classes=10).to(device)
    batch_size = 64

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = efficientnet_b0(num_classes=10).to(device)
    epoch = []
    train_acc = []
    test_acc = []
    learning_rate = 0.03
    num_epochs = 10
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
    #optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    model = train_model(model, loss_fn, train_loader, test_loader, optimizer, num_epochs, device, train_acc, test_acc,
                        epoch)

    plt.figure(figsize=(8, 5))
    plt.plot(epoch, train_acc, label='Training Accuracy', marker='o')
    plt.plot(epoch, test_acc, label='Test accuracy', marker='s')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epoch)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
