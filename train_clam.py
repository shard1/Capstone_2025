import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.model_CLAM import CLAM_SB, CLAM_MB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
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
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

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
                loss_value = loss.item()
            elif coarse: 
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=coarse_gt, instance_eval=True)
                loss = loss_fn(logits, coarse_gt)
                loss_value = loss.item()

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
    


        