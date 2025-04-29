import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),  # Va in paper
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),  # Ua in paper
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        # W_a,m in the paper
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes		not normalized yet
        return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""

class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 subtyping=False, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]  # nn.Linear is W1 in the paper
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)  # attention layers
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK       or Nxn_classes
        A = torch.transpose(A, 1, 0)  # KxN		n_classesxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N			A was not normalized, now it is

        # h is [N,D], A is [n_classes = 1, N] because only one attention branch exists
        M = torch.mm(A, h)  # [n_classes = 1, D]   h_slide
        return M
# class CLAM_SB(nn.Module):
#     def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
#                  instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
#         super().__init__()
#         self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
#         size = self.size_dict[size_arg]
#         fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]  # nn.Linear is W1 in the paper
#         if gate:
#             attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
#         else:
#             attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
#         fc.append(attention_net)
#         self.attention_net = nn.Sequential(*fc)  # attention layers
#         self.classifiers = nn.Linear(size[1], n_classes)  # classifier at the end
#         instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]  # clustering layers
#         self.instance_classifiers = nn.ModuleList(instance_classifiers)
#         self.k_sample = k_sample
#         self.instance_loss_fn = instance_loss_fn
#         self.n_classes = n_classes
#         self.subtyping = subtyping
#
#     @staticmethod
#     def create_positive_targets(length, device):
#         return torch.full((length,), 1, device=device).long()
#
#     @staticmethod
#     def create_negative_targets(length, device):
#         return torch.full((length,), 0, device=device).long()
#
#     # instance-level evaluation for in-the-class attention branch
#     def inst_eval(self, A, h, classifier):
#         device = h.device
#         if len(A.shape) == 1:
#             A = A.view(1, -1)  # change to shape [1,N], N patches so N attention scores	(also one class so [1,N])
#
#         # topk returns top k scores and corresponding indices using attention score A (normalized)
#         # k and B are used interchangeably (B in the paper)
#         top_p_ids = torch.topk(A, self.k_sample)[1][-1]  # topk[1] returns indices of top k values (shape [1 B]) top[1][-1] returns the k indices
#         top_p = torch.index_select(h, dim=0,
#                                    index=top_p_ids)  # selects from h feature vectors using the top k indices, shape [B,D]
#         top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
#         top_n = torch.index_select(h, dim=0,
#                                    index=top_n_ids)  # shape [B, D] because h has shape [N, D] N feature vectors with dimension D
#         p_targets = self.create_positive_targets(self.k_sample, device)  # shape [B]
#         n_targets = self.create_negative_targets(self.k_sample, device)  # shape [B]
#
#         all_targets = torch.cat([p_targets, n_targets], dim=0)  # concatenate p and n targets, so shape is [2B]
#         all_instances = torch.cat([top_p, top_n], dim=0)  # concatenate [2B, D]
#         logits = classifier(all_instances)  # [2B, 2] "logits" corresponds to p_m,k in paper
#         all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)  # shape [2B] returns indices 0,1 (index = class prediction)
#         instance_loss = self.instance_loss_fn(logits, all_targets)
#         return instance_loss, all_preds, all_targets
#
#     # instance-level evaluation for out-of-the-class attention branch
#     def inst_eval_out(self, A, h, classifier):
#         device = h.device
#         if len(A.shape) == 1:
#             A = A.view(1, -1)  # [1, N]
#         top_p_ids = torch.topk(A, self.k_sample)[1][-1]  # [1, B]
#         top_p = torch.index_select(h, dim=0, index=top_p_ids)  # [B, D]
#         p_targets = self.create_negative_targets(self.k_sample, device)  # [B]
#         logits = classifier(top_p)  # [B, 2]
#         p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)  # [B]
#         instance_loss = self.instance_loss_fn(logits, p_targets)
#         return instance_loss, p_preds, p_targets
#
#     def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        # A, h = self.attention_net(h)  # NxK       or Nxn_classes
        # A = torch.transpose(A, 1, 0)  # KxN		n_classes x N
        # if attention_only:
        #     return A
        # A_raw = A
        # A = F.softmax(A, dim=1)  # softmax over N			A was not normalized, now it is
        #
        # if instance_eval:
        #     total_inst_loss = 0.0
        #     all_preds = []
        #     all_targets = []
        #     inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label, label is slide level
        #     for i in range(len(self.instance_classifiers)):
        #         inst_label = inst_labels[i].item()
        #         classifier = self.instance_classifiers[i]
        #         if inst_label == 1:  # in-the-class:
        #             instance_loss, preds, targets = self.inst_eval(A, h, classifier)
        #             all_preds.extend(preds.cpu().numpy())
        #             all_targets.extend(targets.cpu().numpy())
        #         else:  # out-of-the-class
        #             if self.subtyping:
        #                 instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
        #                 all_preds.extend(preds.cpu().numpy())
        #                 all_targets.extend(targets.cpu().numpy())
        #             else:
        #                 continue
        #         total_inst_loss += instance_loss
        #
        #     if self.subtyping:
        #         total_inst_loss /= len(self.instance_classifiers)
        #
        # # h is [N,D], A is [n_classes = 1, N] because only one attention branch exists
        # M = torch.mm(A, h)  # [n_classes = 1, D]   h_slide
        # logits = self.classifiers(M)  # [1, n_classes]
        # Y_hat = torch.topk(logits, 1, dim=1)[1]  # shape [1]		unnormalized logit value
        # Y_prob = F.softmax(logits, dim=1)  # [1, n_classes]		#normalized logit value
        # if instance_eval:
        #     results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
        #                     'inst_preds': np.array(all_preds)}
        # else:
        #     results_dict = {}
        # if return_features:
        #     results_dict.update({'features': M})
        # return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an independent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK     Nxn_classes
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)  # slide level representation h_slide

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict