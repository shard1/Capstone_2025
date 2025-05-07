import argparse
import os

import numpy as np
import torch

from utils import *
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from dataloader.dataloader_AMC import AMCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextWriter():
    def __init__(self, save_path):
        super(TextWriter, self).__init__()
        wf = open(save_path, 'w')
        self.wf = wf

    def print_and_write(self, text):
        print(text)
        self.wf.write(text + "")


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
            cls_mask = (Y == label_class)
            self.data[label_class]["count"] += np.sum(cls_mask)
            self.data[label_class]["correct"] += np.sum(Y_hat[cls_mask] == Y[cls_mask])

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

def summary(model, loader, n_classes):
    acc_logger = AccuracyLogger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, coarse_gt, fine_gt) in enumerate(loader):
        data, coarse_gt, fine_gt = data.to(device), coarse_gt.to(device), fine_gt.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger


def train_clam(epoch, model, loader, optimizer, num_classes, bag_weight, loss_fn=None, hierarchy='coarse-and-fine', log_writer=None):
    model.train()
    is_hierarchy = hierarchy not in ('coarse', 'fine')

    #for hierarchy
    acc_logger_coarse = AccuracyLogger(n_classes=num_classes['coarse'])
    acc_logger_fine = AccuracyLogger(n_classes=num_classes['fine'])
    inst_logger_coarse = AccuracyLogger(n_classes=num_classes['coarse'])
    inst_logger_fine = AccuracyLogger(n_classes=num_classes['fine'])

    #non hierarchy
    acc_logger = AccuracyLogger(n_classes=num_classes[hierarchy])       #tracks bag level accuracy
    inst_logger = AccuracyLogger(n_classes=num_classes[hierarchy])          #tracks instance level accuracy

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    all_preds = []
    all_labels = []

    #for hierarchy
    all_preds_coarse = []
    all_labels_coarse = []
    all_preds_fine = []
    all_labels_fine = []

    log_writer.print_and_write("Training for {} level...".format(hierarchy))

    for batch_idx, (data, coarse_gt, fine_gt) in enumerate(loader):
        data = data.to(device)
        coarse_gt = coarse_gt.to(device)
        fine_gt = fine_gt.to(device)
##################################################################
        if is_hierarchy:
            # coarse and fine
            logits, y_prob, y_hat, _, instance_dict = model(data, num_classes, label=fine_gt, instance_eval=True,
                                                            is_hierarchy=True)
            logits_coarse, logits_fine = logits
            y_prob_coarse, y_prob_fine = y_prob
            y_hat_coarse, y_hat_fine = y_hat

            loss_coarse = loss_fn(logits_coarse, coarse_gt)
            loss_fine = loss_fn(logits_fine, fine_gt)
            loss = loss_coarse + loss_fine

            error = calculate_error(y_hat, fine_gt)
            train_error += error

            acc_logger_coarse.log(y_hat_coarse, coarse_gt)
            acc_logger_fine.log(y_hat_fine, fine_gt)

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger_fine.log_batch(inst_preds, inst_labels)

            all_preds_coarse.append(int(y_hat_coarse))
            all_labels_coarse.append(int(coarse_gt))
            all_preds_fine.append(int(y_hat_fine))
            all_labels_fine.append(int(fine_gt))
############################################################
        else:
            label_gt = fine_gt
            if hierarchy == 'coarse':
                label_gt = coarse_gt
            logits, y_prob, y_hat, _, instance_dict = model(data, num_classes, label=label_gt, instance_eval=True)
            acc_logger.log(y_hat, label_gt)
            loss = loss_fn(logits, label_gt)
            loss_value = loss.item()  # scalar,  bag loss
            
            instance_loss = instance_dict['instance_loss']  # tensor
            inst_count += 1
            instance_loss_value = instance_loss.item()  # scalar, instance loss
            
            train_loss += loss_value  # accumulate total train loss
            train_inst_loss += instance_loss_value  # accumulate instance loss
            total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss  # total loss tensor

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            all_preds.append(int(y_hat))
            all_labels.append(int(label_gt))

            error = calculate_error(y_hat, label_gt)
            train_error += error

        if (batch_idx + 1) % 100 == 0:
            if is_hierarchy:
                log_writer.print_and_write('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx+1,
                                                                                                      loss_value,
                                                                                                      instance_loss_value,
                                                                                                      total_loss.item()) +
                      'label: {}, bag_size: {}'.format(fine_gt.item(), data.size(0)))
                
            else:
                label_gt = coarse_gt if hierarchy == 'coarse' else fine_gt
                
                log_writer.print_and_write('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx+1,
                                                                                                      loss_value,
                                                                                                      instance_loss_value,
                                                                                                      total_loss.item()) +
                      'label: {}, bag_size: {}'.format(label_gt.item(), data.size(0)))
        # backward pass
        total_loss.backward()  # backprop on the entire CLAM model
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)           #bag level loss
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        log_writer.print_and_write('Instance Classifier Stats')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            log_writer.print_and_write('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    log_writer.print_and_write(
        'Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss, train_error))

    if is_hierarchy:
        acc_coarse, acc_fine = 0.0, 0.0
        for i in range(num_classes['coarse']):
            acc_coarse, correct_coarse, count_coarse = acc_logger_coarse.get_summary(i)
            log_writer.print_and_write('Coarse class {}: coarse acc {}, correct coarse {}/{} '.format(i, acc_coarse, correct_coarse,
                                                                                   count_coarse))
        for i in range(num_classes['fine']):
            acc_fine, correct_fine, count_fine = acc_logger_fine.get_summary(i)
            log_writer.print_and_write('Fine class {}: fine acc {}, correct fine {}/{} '.format(i, acc_fine, correct_fine, count_fine))
        
        acc_coarse = accuracy_score(all_labels_coarse, all_preds_coarse)
        acc_fine = accuracy_score(all_labels_fine, all_preds_fine)
        return acc_coarse, acc_fine, train_loss
    else:
        acc = 0.
        for i in range(num_classes[hierarchy]):
            acc, correct, count = acc_logger.get_summary(i)
            log_writer.print_and_write('class {}: acc {}, correct {}/{} '.format(i, acc, correct, count))

        acc = accuracy_score(all_labels, all_preds)
        log_writer.print_and_write(f"Overall Accuracy: {acc:.4f}")
        return acc, train_loss 

def validate_clam(epoch, model, loader, num_classes, loss_fn=None, results_dir=None, hierarchy='coarse-and-fine', log_writer=None):
    model.eval()

    is_hierarchy = hierarchy not in ('coarse', 'fine')
    acc_logger_coarse = AccuracyLogger(n_classes=num_classes['coarse'])
    acc_logger_fine = AccuracyLogger(n_classes=num_classes['fine'])

    acc_logger = AccuracyLogger(n_classes=num_classes[hierarchy])
    inst_logger = AccuracyLogger(n_classes=num_classes[hierarchy])

    val_loss = 0.
    val_error = 0.
    val_inst_loss = 0.
    inst_count = 0

    prob = np.zeros((len(loader), num_classes[hierarchy]))
    prob_coarse = np.zeros((len(loader), num_classes['coarse']))
    prob_fine = np.zeros((len(loader), num_classes['fine']))

    labels = np.zeros(len(loader))
    labels_coarse = np.zeros(len(loader))
    labels_fine = np.zeros(len(loader))

    all_preds_coarse = []
    all_preds_fine = []
    all_labels_coarse = []
    all_labels_fine = []

    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for batch_idx, (data, coarse_gt, fine_gt) in enumerate(loader):
            data, coarse_gt, fine_gt = data.to(device), coarse_gt.to(device), fine_gt.to(device)
        #####################hierarchical################################
            if is_hierarchy:
                logits, y_prob, y_hat, _, instance_dict = model(data, num_classes, label=fine_gt, instance_eval=True,
                                                                is_hierarchy=True)
                logits_coarse, logits_fine = logits
                y_prob_coarse, y_prob_fine = y_prob
                y_hat_coarse, y_hat_fine = y_hat

                acc_logger_coarse.log(y_hat_coarse, coarse_gt)
                acc_logger_fine.log(y_hat_fine, fine_gt)

                loss_coarse = loss_fn(logits_coarse, coarse_gt)
                loss_fine = loss_fn(logits_fine, fine_gt)
                loss = loss_coarse + loss_fine

                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds, inst_labels)

                labels_coarse[batch_idx] = coarse_gt.item()
                labels_fine[batch_idx] = fine_gt.item()
                prob_coarse[batch_idx] = y_prob_coarse.cpu().numpy()
                prob_fine[batch_idx] = y_prob_fine.cpu().numpy()

                all_preds_coarse.append(int(y_hat_coarse))
                all_preds_fine.append(int(y_hat_fine))
                all_labels_coarse.append(int(coarse_gt))
                all_labels_fine.append(int(fine_gt))

                error_coarse = calculate_error(y_hat_coarse, coarse_gt)
                error_fine = calculate_error(y_hat_fine, fine_gt)
                val_error += error_coarse + error_fine
        ######################################################################
            else:
                label_gt = coarse_gt if hierarchy == 'coarse' else fine_gt
                logits, y_prob, y_hat, _, instance_dict = model(data, num_classes, label=label_gt, instance_eval=True)
                acc_logger.log(y_hat, label_gt)
                loss = loss_fn(logits, label_gt)
                val_loss += loss.item()
                
                instance_loss = instance_dict['instance_loss']
                inst_count += 1
                instance_loss_value = instance_loss.item()
                val_inst_loss += instance_loss_value

                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds, inst_labels)
                
                prob[batch_idx] = y_prob.cpu().numpy()
                labels[batch_idx] = label_gt.item()

                all_preds.append(int(y_hat))
                all_labels.append(int(label_gt))

                error = calculate_error(y_hat, label_gt)
                val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)
    #######################hierarchical#########################
    if is_hierarchy:
        auc_coarse = 0
        auc_fine = 0
        if num_classes['coarse'] == 2:
            auc_coarse = roc_auc_score(labels_coarse, prob_coarse[:, 1])
            coarse_auc_vals = []
        else:
            coarse_auc_vals = []
            binary_labels_coarse = label_binarize(labels_coarse, classes=[i for i in range(num_classes['coarse'])])
            for class_idx in range(num_classes['coarse']):
                if class_idx in labels_coarse:
                    fpr, tpr, _ = roc_curve(binary_labels_coarse[:, class_idx], prob_coarse[:, class_idx])
                    coarse_auc_vals.append(calc_auc(fpr, tpr))
                else:
                    coarse_auc_vals.append(float('nan'))

            auc = np.nanmean(np.array(coarse_auc_vals))
        if num_classes['fine'] == 2:
            auc_fine = roc_auc_score(labels_fine, prob_fine[:, 1])
            fine_auc_vals = []
        else:
            fine_auc_vals = []
            binary_labels_fine = label_binarize(labels_fine, classes=[i for i in range(num_classes['fine'])])
            for class_idx in range(num_classes['fine']):
                if class_idx in labels_fine:
                    fpr, tpr, _ = roc_curve(binary_labels_fine[:, class_idx], prob_fine[:, class_idx])
                    fine_auc_vals.append(calc_auc(fpr, tpr))
                else:
                    fine_auc_vals.append(float('nan'))
            auc = np.nanmean(np.array(fine_auc_vals))
        log_writer.print_and_write(
            'Val Set, val_loss: {:.4f}, coarse auc: {:.4f}, fine auc: {:.4f} '.format(val_loss, auc_coarse, auc_fine))
        if inst_count > 0:
            val_inst_loss /= inst_count
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                log_writer.print_and_write('class {} clustering acc {}: correct {}/{} '.format(i, acc, correct, count))
        for i in range(num_classes['coarse']):
            acc, correct, count = acc_logger_coarse.get_summary(i)
            log_writer.print_and_write('class {}: acc {}, correct {}/{} '.format(i, acc, correct, count))
        
        for i in range(num_classes['fine']):
            acc, correct, count = acc_logger_fine.get_summary(i)
            log_writer.print_and_write('class {}: acc {}, correct {}/{} '.format(i, acc, correct, count))
        
        f1_coarse = f1_score(all_labels_coarse, all_preds_coarse, average='macro')
        acc_coarse = accuracy_score(all_labels_coarse, all_preds_coarse)
        f1_fine = f1_score(all_labels_fine, all_preds_fine, average='macro')
        acc_fine = accuracy_score(all_labels_fine, all_preds_fine)
        log_writer.print_and_write(f"Coarse Overall Accuracy: {acc_coarse:.4f}")
        log_writer.print_and_write(f"Coarse Overall F1 Score: {f1_coarse:.4f}")
        log_writer.print_and_write(f"Fine Overall Accuracy: {acc_fine:.4f}")
        log_writer.print_and_write(f"Fine Overall F1 Score: {f1_fine:.4f}")

        return acc_coarse, acc_fine, val_loss
    #############################################################################
    else:
        auc = 0
        if num_classes[hierarchy] == 2:
            auc = roc_auc_score(labels, prob[:, 1])
            auc_vals = []
        else:
            auc_vals = []
            binary_labels = label_binarize(labels, classes=[i for i in range(num_classes[hierarchy])])
            for class_idx in range(num_classes[hierarchy]):
                if class_idx in labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                    auc_vals.append(calc_auc(fpr, tpr))
                else:
                    auc_vals.append(float('nan'))

            auc = np.nanmean(np.array(auc_vals))
        log_writer.print_and_write('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
        
        if inst_count > 0:
            val_inst_loss /= inst_count
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                log_writer.print_and_write('class {} clustering acc {}: correct {}/{} '.format(i, acc, correct, count))
        
        for i in range(num_classes[hierarchy]):
            acc, correct, count = acc_logger.get_summary(i)
            log_writer.print_and_write('class {}: acc {}, correct {}/{} '.format(i, acc, correct, count))

        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        log_writer.print_and_write(f"Overall Accuracy: {acc:.4f}")
        log_writer.print_and_write(f"Overall F1 Score: {f1:.4f}")

        return acc, val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    parser.add_argument('--base_dir', default="/home/user/data/UJSMB_STLB", help='path to dataset')  # [변경] 이미지 패치 저장 경로
    parser.add_argument('--anno_path', default="/home/user/lib/Capstone_2025/dataloader/amc_fine_grained_anno.csv",
                        help='path to dataset')  # [변경] 이미지 패치 저장 경로
    parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')  # [변경]배치 사이즈
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')  # [변경]훈련 반복 수
    parser.add_argument('--epochs_min', default=50, type=int)
    parser.add_argument('--early_stopping_threshold', default=20, type=int)
    parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate',
                        dest='lr')  # [변경] 초기 Learning rate
    # parser.add_argument('--print_freq', default=100, type=int)
    # parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--result', default='./results', type=str, help='path to results')
    parser.add_argument('--bag_loss', default='ce', type=str, help='bag level classifier loss function')
    parser.add_argument('--inst_loss', default='svm', type=str, help='instance classifier loss function')
    parser.add_argument('--model_type', type=str, default='clam_sb', choices=['clam_sb', 'clam_mb'],
                        help='options for a model')
    parser.add_argument('--hierarchy', default='coarse', type=str, choices=['coarse', 'fine', 'coarse-and-fine'],
                        help='choose classification type')
    parser.add_argument('--bag_weight', default=0.7, type=float, help='clam: weight coefficient for bag-level loss')
    args = parser.parse_args()
    args.result = os.path.join(args.result, args.hierarchy)
    os.makedirs(args.result, exist_ok=True)

    set_seed(args.seed)
    log_writer = TextWriter(os.path.join(args.result, '{} log.txt'.format(args.model_type)))

    log_writer.print_and_write("Preparing data...")

    train_dataset = AMCDataset(args.base_dir, args.anno_path, split="train")
    val_dataset = AMCDataset(args.base_dir, args.anno_path, split="val")
    test_dataset = AMCDataset(args.base_dir, args.anno_path, split="test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=identity_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)

    log_writer.print_and_write("Training on {} samples".format(len(train_dataset)))
    log_writer.print_and_write("Validation on {} samples".format(len(val_dataset)))
    log_writer.print_and_write("Test on {} samples".format(len(test_dataset)))

    log_writer.print_and_write("Preparing model...")
    model_args = {"gate": True, "size_arg": "small", "dropout": 0.25, "k_sample": 8}
    class_dict = {'coarse': 4, 'fine': 14, 'coarse-and-fine': 14}


    model, loss_fn = configure_clam(model_args, args.model_type, hierarchy=args.hierarchy, bag_loss=args.bag_loss, inst_loss=args.inst_loss)
    model = model.to(device)
    
    log_writer.print_and_write("Done")

    log_writer.print_and_write("Setting optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    log_writer.print_and_write("Done")

    train_accs = []
    val_accs = []
    train_losses = []
    validation_losses = []

    best_acc, best_epochs = 0, 0
    best_save_path = os.path.join(args.result, "best.pth")
    for epoch in range(args.epochs):
        if args.hierarchy in ["fine", "coarse"]:
            train_acc, train_loss = train_clam(epoch+1, model, train_loader, optimizer, class_dict,
                                    bag_weight=args.bag_weight, loss_fn=loss_fn,
                                    hierarchy=args.hierarchy, log_writer=log_writer)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            val_acc, val_loss = validate_clam(epoch+1, model, val_loader, class_dict, loss_fn=loss_fn,
                                    hierarchy=args.hierarchy, log_writer=log_writer)
            validation_losses.append(val_loss)
            val_accs.append(val_acc)
        else:       #for hierarchy
            raise NotImplementedError

        if best_acc < val_acc:
            best_acc, best_epochs = val_acc, epoch
            save_path = os.path.join(args.result, "{}.pth".format(epoch+1))
            torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), best_save_path)

        # Early stopping
        if epoch - best_epochs > args.early_stopping_threshold and epoch > args.epochs_min:
            break

    drawPlot(train_losses, args.model_type, args.hierarchy, save_path=os.path.join(args.result, "train_loss.png"), label="Train loss")
    drawPlot(validation_losses, args.model_type, args.hierarchy, save_path=os.path.join(args.result, "validation_loss.png"), label="Validation loss")
    drawPlot(train_accs, args.model_type, args.hierarchy, save_path=os.path.join(args.result, "train_acc.png"), label="Train accuracy")
    drawPlot(val_accs, args.model_type, args.hierarchy, save_path=os.path.join(args.result, "validation_acc.png"), label="Validation accuracy")

    model = model.load_state_dict(torch.load(best_save_path))

    _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes)
    log_writer.print_and_write('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    log_writer.print_and_write('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
