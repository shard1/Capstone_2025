from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
from utils import *
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from dataloader.dataloader_AMC import AMCDataset
from models.HPMIL import HPMIL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mapping = torch.zeros(11,4)

mapping[0][0] = 1
mapping[1][0] = 1
mapping[2][1] = 1
mapping[3][1] = 1
mapping[4][1] = 1
mapping[5][2] = 1
mapping[6][2] = 1
mapping[7][3] = 1
mapping[8][3] = 1
mapping[9][3] = 1
mapping[10][3] = 1


class TextWriter():
    def __init__(self, save_path):
        super(TextWriter, self).__init__()
        wf = open(save_path, 'w')
        self.wf = wf

    def print_and_write(self, text):
        print(text)
        self.wf.write(text + "\n")

def compute_losses(logits_coarse, logits_fine, y_coarse, y_fine, mapping):
    loss_fn = nn.CrossEntropyLoss()
    loss_coarse = loss_fn(logits_coarse, y_coarse)
    loss_fine = loss_fn(logits_fine, y_fine)

    p_fine = F.softmax(logits_fine, dim=-1)
    p_coarse = F.softmax(logits_coarse, dim=-1)

    p_fine_grouped = p_fine @ mapping.to(p_fine.device)
    loss_kl = F.kl_div(p_fine_grouped.log(), p_coarse, reduction='batchmean')

    return loss_coarse, loss_fine, loss_kl

def train(model, train_loader, optimizer, mapping, num_coarse, num_fine, lambda_kl = 0.2, log_writer=None):
    model.train()
    train_loss = 0
    preds_c = []
    preds_f = []
    labels_c = [] 
    labels_f = []
    log_writer.print_and_write("Training for {} coarse classes and {} fine classes...".format(num_coarse, num_fine))
    for batch_idx, batch in enumerate(train_loader):
        data, y_coarse, y_fine = batch
        data = data.unsqueeze(0).to(device)
        y_coarse = y_coarse.to(device)
        y_fine = y_fine.to(device)

        logits_coarse, logits_fine = model(data)
        loss_c, loss_f, loss_kl = compute_losses(logits_coarse, logits_fine, y_coarse, y_fine, mapping)
        total_loss = loss_c + loss_f + lambda_kl*loss_kl
        loss_value = loss_c.item() + loss_f.item()
        train_loss += loss_value

        pred_coarse = torch.topk(logits_coarse, 1, dim=1)[1]
        pred_fine = torch.topk(logits_fine, 1, dim=1)[1]

        preds_c.append(int(pred_coarse))
        preds_f.append(int(pred_fine))
        labels_c.append(int(y_coarse))
        labels_f.append(int(y_fine))

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if (batch_idx + 1) % 100 == 0:
            log_writer.print_and_write('batch {}, loss: {:.4f}, weighted loss: {:.4f} '.format(batch_idx+1, loss_value, total_loss.item()) +
                        'label: {}, bag_size: {}'.format(y_fine.item(), data.size(1)))
                
    
    train_loss /= len(train_loader)
    acc_coarse = accuracy_score(preds_c, labels_c)
    acc_fine = accuracy_score(labels_f, preds_f)
    log_writer.print_and_write('\nEpoch: {}, train_loss: {:.4f}, Coarse Acc: {:.4f}, Fine Acc: {:.4f}'.format(epoch, train_loss, acc_coarse, acc_fine))
    return acc_coarse, acc_fine, train_loss

def validate(model, val_loader, mapping, lambda_kl=0.2, log_writer=None):
    model.eval()
    preds_c = []
    preds_f = []
    labels_c = [] 
    labels_f = []
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            data, y_coarse, y_fine = batch
            data = data.unsqueeze(0).to(device)
            y_coarse = y_coarse.to(device)
            y_fine = y_fine.to(device)

            logits_coarse, logits_fine = model(data)
            pred_coarse = torch.topk(logits_coarse, 1, dim=1)[1]
            pred_fine = torch.topk(logits_fine, 1, dim=1)[1]
            preds_c.append(int(pred_coarse))
            preds_f.append(int(pred_fine))

            labels_c.append(int(y_coarse))
            labels_f.append(int(y_fine))

            loss_coarse, loss_fine, loss_kl = compute_losses(logits_coarse, logits_fine, y_coarse, y_fine, mapping)
            val_loss += loss_coarse.item() + loss_fine.item()
    acc_coarse = accuracy_score(labels_c, preds_c)
    acc_fine = accuracy_score(labels_f, preds_f)
    f1_coarse = f1_score(labels_c, preds_c, average='macro')
    f1_fine = f1_score(labels_f, preds_f, average='macro')
    log_writer.print_and_write('Val Set, val_loss: {:.4f}'.format(val_loss))
    
    log_writer.print_and_write(f"Coarse Overall Accuracy: {acc_coarse:.4f}")
    log_writer.print_and_write(f"Coarse Overall F1 Score: {f1_coarse:.4f}")
    log_writer.print_and_write(f"Fine Overall Accuracy: {acc_fine:.4f}")
    log_writer.print_and_write(f"Fine Overall F1 Score: {f1_fine:.4f}\n")
    return acc_coarse, acc_fine, f1_coarse, f1_fine, val_loss/len(val_loader)

def test(model, test_loader, num_coarse, num_fine):
    model.eval()
    all_preds_coarse = []
    all_preds_fine = []
    all_labels_coarse = []
    all_labels_fine = []

    with torch.no_grad():
        for batch in test_loader:
            data, y_coarse, y_fine = batch
            data = data.unsqueeze(0).to(device)
            y_coarse = y_coarse.to(device).squeeze(0)
            y_fine = y_fine.to(device).squeeze(0)

            logits_coarse, logits_fine = model(data)
            pred_coarse = torch.topk(logits_coarse, 1, dim=1)[1]
            pred_fine = torch.topk(logits_fine, 1, dim=1)[1]

            all_preds_coarse.append(int(pred_coarse))
            all_preds_fine.append(int(pred_fine))
            all_labels_coarse.append(int(y_coarse))
            all_labels_fine.append(int(y_fine))
    acc_coarse = accuracy_score(all_labels_coarse, all_preds_coarse)
    acc_fine = accuracy_score(all_labels_fine, all_preds_fine)
    f1_coarse = f1_score(all_labels_coarse, all_preds_coarse, average='macro')
    f1_fine = f1_score(all_labels_fine, all_preds_fine, average='macro')

    return acc_coarse, acc_fine, f1_coarse, f1_fine

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
    parser.add_argument('--epochs_min', default=10, type=int)
    parser.add_argument('--early_stopping_threshold', default=20, type=int)
    parser.add_argument('--lr', default=1e-6, type=float, help='initial learning rate', dest='lr')  # [변경] 초기 Learning rate
    # parser.add_argument('--print_freq', default=100, type=int)
    # parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--result', default='./results', type=str, help='path to results')
    parser.add_argument('--bag_loss', default='ce', type=str, help='bag level classifier loss function')
    parser.add_argument('--inst_loss', default='svm', type=str, help='instance classifier loss function')
    parser.add_argument('--model_type', type=str, default='clam_mb', choices=['clam_sb', 'clam_mb', 'hpmil'],
                        help='options for a model')
    parser.add_argument('--hierarchy', default='coarse_and_fine', type=str, choices=['coarse', 'fine', 'coarse_and_fine'],
                        help='choose classification type')
    
    # parser.add_argument('--bag_weight', default=0.7, type=float, help='clam: weight coefficient for bag-level loss')
    parser.add_argument('--lr_str', default='1e-6', type=str)
    parser.add_argument('--mode', type = str, default = 'train', choices = ['train', 'test'])
    parser.add_argument('--min_patch', default=8, type=int, help = 'min number of top k patches for inst eval')
    args = parser.parse_args()

    args.result = os.path.join(args.result, args.model_type, args.hierarchy, args.lr_str)
    os.makedirs(args.result, exist_ok=True)

    set_seed(args.seed)
    log_writer = TextWriter(os.path.join(args.result, '{} log.txt'.format(args.model_type)))

    log_writer.print_and_write("Preparing data...")

    train_dataset = AMCDataset(args.base_dir, args.anno_path, split="train", min_patches=args.min_patch)
    val_dataset = AMCDataset(args.base_dir, args.anno_path, split="val", min_patches=args.min_patch)
    test_dataset = AMCDataset(args.base_dir, args.anno_path, split="test", min_patches=args.min_patch)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=identity_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)

    train_coarse_accs = []
    train_fine_accs = []
    val_coarse_accs = []
    val_fine_accs = []
    f1_coarse = []
    f1_fine = []
    validation_losses = []

    train_losses = []
    train_coarse_accs = []
    train_fine_accs = []

    class_dict = {'coarse': 4, 'fine': 11, 'coarse_and_fine': 15}
    best_acc, best_epochs = 0, 0
    best_save_path = os.path.join(args.result, "best.pth")

    log_writer.print_and_write("Training on {} samples".format(len(train_dataset)))
    log_writer.print_and_write("Validation on {} samples".format(len(val_dataset)))
    log_writer.print_and_write("Test on {} samples".format(len(test_dataset)))

    log_writer.print_and_write("Preparing model...")


    coarse_proto = torch.load("/home/user/lib/Capstone_2025/prototypes/coarse_proto.pt")
    fine_proto = torch.load("/home/user/lib/Capstone_2025/prototypes/fine_proto.pt")
    model = HPMIL(class_dict['coarse'], class_dict['fine'], coarse_proto, fine_proto, dropout = 0.1)

    model = model.to(device)
    log_writer.print_and_write("Done")

    log_writer.print_and_write("Setting optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    log_writer.print_and_write("Done")

    if args.mode == 'train':
        for epoch in range(args.epochs):
        
            acc_coarse, acc_fine, train_loss = train(model, train_loader, optimizer, mapping, class_dict['coarse'], 
                                                     class_dict['fine'], lambda_kl=0.2, log_writer=log_writer)
            train_losses.append(train_loss)
            train_coarse_accs.append(acc_coarse)
            train_fine_accs.append(acc_fine)

            val_acc_coarse, val_acc_fine, val_f1_coarse, val_f1_fine, val_loss = validate(model, val_loader, mapping, lambda_kl=0.2,
                                   log_writer=log_writer)
            validation_losses.append(val_loss)
            val_coarse_accs.append(val_acc_coarse)
            val_fine_accs.append(val_acc_fine)
            f1_fine.append(val_f1_fine)
            f1_coarse.append(val_f1_coarse)

            if best_acc < val_acc_fine:
                best_acc, best_epochs = val_acc_fine, epoch
                save_path = os.path.join(args.result, "{}.pth".format(epoch+1))
                torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), best_save_path)

            # Early stopping
            if epoch - best_epochs > args.early_stopping_threshold and epoch > args.epochs_min:
                break

   
        drawPlot(train_losses, args.model_type, args.hierarchy, save_path=os.path.join(args.result, "train_loss.png"),
                    label="Train loss")
        drawPlot(validation_losses, args.model_type, args.hierarchy,
                    save_path=os.path.join(args.result, "validation_loss.png"), label="Validation loss")
        drawPlot(f1_coarse, args.model_type, args.hierarchy, save_path=os.path.join(args.result, "f1_coarse.png"),
                    label="f1 Score Coarse")
        drawPlot(f1_fine, args.model_type, args.hierarchy, save_path=os.path.join(args.result, "f1_fine.png"),
                    label="f1 Score Fine")
        drawPlot(train_coarse_accs, args.model_type, args.hierarchy, save_path=os.path.join(args.result, "train_coarse_acc.png"),
                    label="Train Coarse accuracy")
        drawPlot(val_coarse_accs, args.model_type, args.hierarchy, save_path=os.path.join(args.result, "validation_coarse_acc.png"),
                    label="Validation Coarse accuracy")
        drawPlot(train_fine_accs, args.model_type, args.hierarchy,
                    save_path=os.path.join(args.result, "train_fine_acc.png"),
                    label="Train Fine accuracy")
        drawPlot(val_fine_accs, args.model_type, args.hierarchy,
                    save_path=os.path.join(args.result, "validation_fine_acc.png"),
                    label="Validation Fine accuracy")
        model.load_state_dict(torch.load(best_save_path))

    log_writer.print_and_write("Evaluation on Test Set...")
    f1_coarse, f1_fine, acc_coarse, acc_fine = test(model, test_loader, class_dict['coarse'], class_dict['fine'])
    log_writer.print_and_write('f1 Coarse: {:.4f}, f1 Fine: {:.4f}, Coarse Acc: {:.4f}, Fine Acc: {:.4f}'.format(f1_coarse, f1_fine, acc_coarse, acc_fine))