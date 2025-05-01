import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from tensorboardX import SummaryWriter

from models.model_CLAM import CLAM_SB, CLAM_MB
from dataloader.dataloader_AMC import AMCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AccuracyLogger(object):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()
    
    def initialize(self):
        self.data = [{"count" : 0, "correct" : 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
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


def train_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, coarse = True, fine = True):
    model.train()
    acc_logger = AccuracyLogger(n_classes=n_classes)
    inst_logger = AccuracyLogger(n_classes=n_classes)

    train_loss = 0.
    inst_loss = 0.
    inst_count = 0

    if coarse or fine:
        for i, (data, coarse_gt, fine_gt) in enumerate(loader):
            data = (torch.load(data)).to(device)
            coarse_gt = coarse_gt.to(device)
            fine_gt = fine_gt.to(device)
            if coarse and fine:
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=coarse_gt, instance_eval=True)
                loss = loss_fn(logits, coarse_gt)

            elif coarse:
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=coarse_gt, instance_eval=True)
                loss = loss_fn(logits, coarse_gt)

            elif fine: 
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=fine_gt, instance_eval=True)      
                loss = loss_fn(logits, fine_gt)
            
            loss_value = loss.item()
            instance_loss = instance_dict['instance_loss']
            inst_count += 1
            instance_loss_value = instance_loss.item()
            
            train_inst_loss += instance_loss_value
            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss
            
            #for logging
            inst_preds = instance_dict['inst_pred']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            train_loss += loss_value
            if (i + 1) % 20 == 0:
                print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                    'label: {}, bag_size: {}'.format(coarse_gt.item(), data.size(0)))

            # error = calculate_error(Y_hat, label)
            # train_error += error
            
            # backward pass
            total_loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()

            # calculate loss and error for epoch
            train_loss /= len(loader)
            #train_error /= len(loader)

            if inst_count > 0:
                train_inst_loss /= inst_count
                print('\n')
                for i in range(2):
                    acc, correct, count = inst_logger.get_summary(i)
                    print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

            print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
            for i in range(n_classes):
                acc, correct, count = acc_logger.get_summary(i)
                print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
                if writer and acc is not None:
                    writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

            if writer:
                writer.add_scalar('train/loss', train_loss, epoch)
                #writer.add_scalar('train/error', train_error, epoch)
                writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

    else:
        pass  #raise error
    
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
    parser.add_argument('--inst_loss', default='svm', type=str, help='path to checkpoint')
    parser.add_argument('--model_type', type=str, choices = ['clam_sb', 'clam_mb'], help = 'options for a model')
    args = parser.parse_args()

    # writer_dir = os.path.join(args.results_dir, str(cur))
    # if not os.path.isdir(writer_dir):
    #     os.mkdir(writer_dir)
    #
    # if args.log_data:
    #     from tensorboardX import SummaryWriter
    #
    #     writer = SummaryWriter(writer_dir, flush_secs=15)
    #
    # else:
    #     writer = None

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


    print("\nChoosing loss function...\n")
    if args.inst_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else :
        loss_fn = nn.CrossEntropyLoss()

    print("\nPreparing model...\n")
    model_args = {"gate" : True, "size_arg" : "small", "dropout" : 0.25,
                  "k_sample" : 8, "n_classes" : 2}

    if args.model_type == 'clam_mb':
        model = CLAM_MB(**model_args, instance_loss_fn = loss_fn, subtyping=True)
    elif args.model_type == 'clam_sb':
        model = CLAM_SB(**model_args, instance_loss_fn=loss_fn, subtyping=True)
    else:
        raise NotImplementedError


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


        