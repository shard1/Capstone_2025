import argparse
import os
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader.dataloader_AMC import AMCDataset
from models import HiClass

def train(args, epoch, model, loss_fn, train_loader, optimizer, device):
    model.train()

    train_loss = 0.0
    correct_coarse = 0
    correct_fine = 0
    train_n = 0

    for i, (data, coarse_gt, fine_gt) in enumerate(train_loader):
        patch_feature = (torch.load(data)).to(device)
        coarse_gt = coarse_gt.to(device)
        fine_gt = fine_gt.to(device)

        pred_coarse, pred_fine = HiClass(patch_feature)

        loss_ce = nn.CrossEntropyLoss()
        #loss_con = nn.
        loss_coarse = loss_ce + loss_con + loss_kl + loss_gce
        loss_fine = loss_ce + loss_con + loss_kl + loss_gce
        
        optimizer.zero_grad()
        #loss.backward()
        optimizer.step()

        #_, pred = torch.max(outputs, 1)
        correct_coarse += torch.sum(pred_coarse == coarse_gt).item()
        correct_fine += torch.sum(pred_fine == fine_gt).item()
        train_n += coarse_gt.size(0)
        #train_loss += loss.item()

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: %d | %d | Loss: %.4f | Coarse Acc: %.2f%% | Fine Acc: %.2f%%' % (epoch + 1, i, train_loss / train_n,
                                                                       (correct_coarse / train_n) * 100),
                                                                       (correct_fine/train_n) * 100)

    return train_corr / train_n, train_loss / train_n


def test(args, epoch, model, test_loader, device, tag="Validation"):
    model.eval()

    test_corr_coarse, test_corr_fine, test_n = 0, 0, 0
    with torch.no_grad():
        for i, (data, coarse_gt, fine_gt) in enumerate(test_loader):
            patch_feature = (torch.load(data)).to(device)
            coarse_gt = coarse_gt.to(device)
            fine_gt = fine_gt.to(device)

            output_coarse, output_fine = HiClass(patch_feature)

            _, pred_coarse = torch.max(output_coarse, 1)
            _, pred_fine = torch.max(output_fine, 1)

            test_corr_coarse += torch.sum(pred_coarse == coarse_gt).item()
            test_corr_fine += torch.sum(pred_fine == fine_gt).item()
            test_n += coarse_gt.size(0)

            if i % args.print_freq == 0 or i == len(train_loader) - 1:
                test_acc_coarse = (test_corr_coarse / test_n) * 100
                test_acc_fine = (test_corr_fine / test_n) * 100
                print("Epoch: {} | {} | {} Coarse Acc: {:.4f} Fine Acc: {:.4f}".format(
                    epoch, i, tag, test_acc_coarse, test_acc_fine))

    return test_acc_coarse, test_acc_fine


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    parser.add_argument('--data', default="/home/user/data/UJSMB_STLB", help='path to dataset')  # [변경] 이미지 패치 저장 경로
    parser.add_argument('--anno', default="/home/user/lib/Capstone_2025/dataloader/amc_fine_grained_anno.csv",
                        help='path to dataset')  # [변경] 이미지 패치 저장 경로
    parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')  # [변경]배치 사이즈
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')  # [변경]훈련 반복 수
    parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate',
                        dest='lr')  # [변경] 초기 Learning rate
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--result', default='./results', type=str, help='path to results')
    args = parser.parse_args()

    os.makedirs(args.result, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    train_dataset = AMCDataset(args.data, args.anno, split="train")
    val_dataset = AMCDataset(args.data, args.anno, split="val")
    test_dataset = AMCDataset(args.data, args.anno, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # TO DO
    model =
    loss_fn =
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_acc, train_loss = train(args, epoch, model, loss_fn, train_loader, optimizer, device)
        val_acc_coarse, val_acc_fine = test(args, epoch, model, val_loader, device, tag="Validation")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.result, "best_model.pth"))

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.result, "model_{}.pth".format(epoch)))

    test_acc_coarse, test_acc_fine = test(args, epoch, model, test_loader, device, tag="Test")
