import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from topk.svm import SmoothTop1SVM

from models.model_CLAM import CLAM_SB, CLAM_MB
from dataloader.dataloader_AMC import AMCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AccuracyLogger(object):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)  # predictions for a batch
        Y = np.array(Y).astype(int)  # ground truth for a batch
        for label_class in np.unique(Y):  # unique class labels in a batch, no duplicates
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


def configure_loss_fns(bag_loss, inst_loss, n_classes):
    if bag_loss == 'svm':
        loss_fn = SmoothTop1SVM(n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()

    if inst_loss == 'svm':
        instance_loss_fn = SmoothTop1SVM(n_classes)
        if device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()
    return loss_fn, instance_loss_fn


def configure_model(model_args, model_type, inst_loss_fn, n_classes):
    if model_type == 'clam_mb':
        model = CLAM_MB(**model_args, n_classes=n_classes, instance_loss_fn=inst_loss_fn, subtyping=True)
    elif args.model_type == 'clam_sb':
        model = CLAM_SB(**model_args, instance_loss_fn=inst_loss_fn, subtyping=True)
    else:
        raise NotImplementedError
    return model


def configure_clam(model_args, model_type, hierarchy, bag_loss, inst_loss):
    if hierarchy == 'coarse-only':
        loss_fn, instance_loss_fn = configure_loss_fns(bag_loss, inst_loss, n_classes=4)
        model = configure_model(model_args, model_type, instance_loss_fn, 4)
    elif hierarchy == 'fine-only':
        loss_fn, instance_loss_fn = configure_loss_fns(bag_loss, inst_loss, n_classes=14)
        model = configure_model(model_args, model_type, instance_loss_fn, 14)
    else:
        loss_fn, instance_loss_fn = configure_loss_fns(bag_loss, inst_loss, n_classes=18)
        model = configure_model(model_args, model_type, instance_loss_fn, 18)
    return model, loss_fn


def train_clam(epoch, model, loader, optimizer, num_classes, bag_weight, loss_fn=None, hierarchy='coarse-and-fine'):
    model.train()

    is_hierarchy = hierarchy not in ('coarse', 'fine')
    if is_hierarchy:
        acc_logger_coarse = AccuracyLogger(n_classes = num_classes['coarse'])
        acc_logger_fine = AccuracyLogger(n_classes = num_classes['fine'])
        inst_logger_coarse = AccuracyLogger(n_classes = num_classes['coarse'])
        inst_logger_fine = AccuracyLogger(n_classes = num_classes['fine'])
    else:
        acc_logger = AccuracyLogger(n_classes=num_classes[hierarchy])
        inst_logger = AccuracyLogger(n_classes=num_classes[hierarchy])

    train_loss = 0.
    # train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print("Training for {} level...\n".format(hierarchy))
    for batch_idx, (data, coarse_gt, fine_gt) in enumerate(loader):
        data = (torch.load(data)).to(device)
        coarse_gt = coarse_gt.to(device)
        fine_gt = fine_gt.to(device)

        if hierarchy == 'coarse':
            logits, y_prob, y_hat, _, instance_dict = model(data, label=coarse_gt, instance_eval=True)
            loss = loss_fn(logits, coarse_gt)
            acc_logger.log(y_hat, coarse_gt)
            inst_preds = instance_dict['inst_pred']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)
        elif hierarchy == 'fine':
            logits, y_prob, y_hat, _, instance_dict = model(data, label=fine_gt, instance_eval=True)
            loss = loss_fn(logits, fine_gt)
            acc_logger.log(y_hat, fine_gt)
            inst_preds = instance_dict['inst_pred']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)
        else:
            combined = coarse_gt + fine_gt
            logits, y_prob, y_hat, _, instance_dict = model(data, label=combined, instance_eval=True, is_hierarchy=True)
            logits_coarse, logits_fine = logits
            y_prob_coarse, y_prob_fine = y_prob
            y_hat_coarse, y_hat_fine = y_hat
            
            loss_coarse = loss_fn(logits_coarse, coarse_gt)
            loss_fine = loss_fn(logits_fine, fine_gt)
            loss = loss_coarse + loss_fine
            
            acc_logger_coarse.log(y_hat_coarse, coarse_gt)
            acc_logger_fine.log(y_hat_fine, fine_gt)
            
            
            inst_preds = instance_dict['inst_pred']
            inst_labels = instance_dict['inst_labels']
            inst_logger_coarse.log_batch(inst_preds, inst_labels)
            inst_logger_fine.log_batch()

        loss_value = loss.item()  # scalar,  bag loss
        instance_loss = instance_dict['instance_loss']  # tensor
        inst_count += 1
        instance_loss_value = instance_loss.item()  # scalar, instance loss

        train_inst_loss += instance_loss_value  # accumulate instance loss
        train_loss += loss_value  # accumulate total train loss
        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss  # total loss tensor

        if (batch_idx + 1) % 20 == 0:
            if hierarchy == 'coarse-only':
                print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx,
                                                                                                      loss_value,
                                                                                                      instance_loss_value,
                                                                                                      total_loss.item()) +
                      'label: {}, bag_size: {}'.format(coarse_gt.item(), data.size(0)))
            elif hierarchy == 'fine-only':
                print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx,
                                                                                                      loss_value,
                                                                                                      instance_loss_value,
                                                                                                      total_loss.item()) +
                      'label: {}, bag_size: {}'.format(fine_gt.item(), data.size(0)))
            else:
                pass

        # error = calculate_error(Y_hat, label)
        # train_error += error

        # backward pass
        total_loss.backward()  # backprop on the entire CLAM model
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    # train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    # print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        if is_hierarchy:
            acc, correct, count = acc_logger_coarse.get_summary(i)
            print('Coarse class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            acc, correct, count = acc_logger_fine.get_summary(i)
            print('Fine class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        else:
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    parser.add_argument('--base_dir', default="/home/user/data/UJSMB_STLB", help='path to dataset')  # [변경] 이미지 패치 저장 경로
    parser.add_argument('--anno_path', default="/home/user/lib/Capstone_2025/dataloader/amc_fine_grained_anno.csv",
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
    parser.add_argument('--bag_loss', default='svm', type=str, help='bag level classifier loss function')
    parser.add_argument('--inst_loss', default='svm', type=str, help='instance classifier loss function')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb'], help='options for a model')
    parser.add_argument('--hierarchy', default='coarse-and-fine', type=str, choices=['coarse', 'fine', 'coarse-and-fine'],
                        help='choose classification type')
    parser.add_argument('--bag_weight', default=0.7, type=float, help='clam: weight coefficient for bag-level loss')
    args = parser.parse_args()

    print("Preparing data...\n")

    train_dataset = AMCDataset(args.base_dir, args.anno_path, split="train")
    val_dataset = AMCDataset(args.base_dir, args.anno_path, split="val")
    test_dataset = AMCDataset(args.base_dir, args.anno_path, split="test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Training on {} samples\n".format(len(train_dataset)))
    print("Validation on {} samples\n".format(len(val_dataset)))
    print("Test on {} samples\n".format(len(test_dataset)))

    print("\nPreparing model...\n")
    model_args = {"gate": True, "size_arg": "small", "dropout": 0.25,
                  "k_sample": 8}
    class_dict = {'coarse' : 4, 'fine' : 14, 'coarse-and-fine' : 18}

    model, loss_fn = configure_clam(model_args, args.model_type, args.hierarchy, args.bag_loss, args.inst_loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        if args.model_type in ['clam_mb', 'clam_sb'] and args.hierarchy in ['coarse', 'fine', 'coarse-and-fine']:
            if args.hierarchy == 'coarse':
                train_clam(epoch, model, train_loader, optimizer, class_dict, n_classes=4, bag_weight=args.bag_weight,
                           loss_fn=loss_fn, hierarchy=args.hierarchy)
            elif args.hierarchy == 'fine':
                train_clam(epoch, model, train_loader, optimizer, class_dict, n_classes=14, bag_weight=args.bag_weight,
                           loss_fn=loss_fn, hierarchy=args.hierarchy)
            else:
                train_clam(epoch, model, train_loader, optimizer, class_dict, n_classes=18, bag_weight=args.bag_weight,
                           loss_fn=loss_fn, hierarchy=args.hierarchy)
        else:
            pass

    # _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes)
    # print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    #
    # results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    # print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))