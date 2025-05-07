from models.model_CLAM import CLAM_SB, CLAM_MB
from topk.svm import SmoothTop1SVM
import matplotlib.pyplot as plt
import torch.nn as nn
import random

def configure_loss_fns(bag_loss, inst_loss, n_classes):
    if bag_loss == 'svm':
        loss_fn = SmoothTop1SVM(n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()

    if inst_loss == 'svm':
        instance_loss_fn = SmoothTop1SVM(2)
        if device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()
    return loss_fn, instance_loss_fn


def configure_model(model_args, model_type, inst_loss_fn, n_classes):
    if model_type == 'clam_mb':
        model = CLAM_MB(**model_args, n_classes=n_classes, instance_loss_fn=inst_loss_fn, subtyping=True)
    elif model_type == 'clam_sb':
        model = CLAM_SB(**model_args, n_classes=n_classes, instance_loss_fn=inst_loss_fn, subtyping=True)
    else:
        raise NotImplementedError
    return model


def configure_clam(model_args, model_type, hierarchy, bag_loss, inst_loss):
    if hierarchy == 'coarse':
        loss_fn, instance_loss_fn = configure_loss_fns(bag_loss, inst_loss, n_classes=4)
        model = configure_model(model_args, model_type, instance_loss_fn, n_classes=4)
    elif hierarchy == 'fine':
        loss_fn, instance_loss_fn = configure_loss_fns(bag_loss, inst_loss, n_classes=14)
        model = configure_model(model_args, model_type, instance_loss_fn, 14)
    else:
        loss_fn, instance_loss_fn = configure_loss_fns(bag_loss, inst_loss, n_classes=14)
        model = configure_model(model_args, model_type, instance_loss_fn, 14)
    return model, loss_fn

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error

def identity_collate(batch):
    return batch[0]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    

def drawPlot(acc, model_type, hierarchy, save_path=None, label=""):
    plot_title = "{} ({}_{})".format(label, hierarchy, model_type)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(acc) + 1), acc, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.title(plot_title)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
