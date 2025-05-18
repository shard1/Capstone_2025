import torch
import torch.nn as nn
import torch.nn.functional as F
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=True, n_classes=1):
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


class PrototypeBlock(nn.Module):
    def __init__(self, k, d, similarity = 'dot'):
        super().__init__(PrototypeBlock, self)
        self.prototypes = nn.Parameter(torch.randn(k, d))
        self.similarity = similarity
    def forward(self, x):
        if self.similarity == 'cosine':
            x = F.normalize(x, dim=-1)
            p = F.normalize(self.prototypes, dim=-1)
        else:
            p = self.prototypes
        sim = torch.einsum('bnd, kd->bnk', x, p)
        return sim
class HPMIL(nn.Module):
    def __init__(self, k_coarse, k_fine, gate=True, size_arg="small", dropout=0.,
                 subtyping=False, embed_dim=1024):
        super(HPMIL, self).__init__()
        self.k_coarse = k_coarse
        self.k_fine = k_fine
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)] 
        self.classifiers = 
        

