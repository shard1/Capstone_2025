import argparse
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0

from models.vit_vanilla import ViTVanilla
from dataloader.dataloader_AMC import AMCDataset


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

            if i % 100 == 0:
                print('Epoch: %d | %d | Loss: %.4f | Train Acc: %.2f%%' % (
                epoch + 1, i, train_loss / train_n, (train_corr / train_n) * 100))

        print('Epoch: %d | Loss: %.4f | Train Acc: %.2f%%' % (
            epoch + 1, train_loss / train_n, (train_corr / train_n) * 100))
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
    # Arguments 설정
    # parser = argparse.ArgumentParser(description='PyTorch Training')
    # parser.add_argument('--data', default='./Data/Qupath2/patch', help='path to dataset')  # [변경] 이미지 패치 저장 경로
    # parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    # parser.add_argument('--input_size', default=512, type=int, help='image input size')  # [변경] 입력 이미지의 크기
    # parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    # parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')  # [변경]훈련 반복 수
    # parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')  # [변경]배치 사이즈
    # parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate', dest='lr')  # [변경] 초기 Learning rate
    # parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    # parser.add_argument('--result', default='results_ver1', type=str, help='path to results_ver1')
    # args = parser.parse_args()
    #
    # if args.seed is not None:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = True
    #
    # train_dataset = AMCDataset(base_dir, anno_path, split="train")
    # val_dataset = AMCDataset(base_dir, anno_path, split="val")
    # test_dataset = AMCDataset(base_dir, anno_path, split="test")
    #
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)






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

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = efficientnet_b0(num_classes=10).to(device)
    # model = ViTVanilla(num_classes=10).to(device)
    epoch = []
    train_acc = []
    test_acc = []
    learning_rate = 0.001
    num_epochs = 10
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    model = train_model(model, loss_fn, train_loader, test_loader, optimizer, num_epochs, device, train_acc, test_acc,
                        epoch)

    plt.figure(figsize=(8, 5))
    plt.plot(epoch, train_acc, label='Training accuracy', marker='o')
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



