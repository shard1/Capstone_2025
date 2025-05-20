import torch
import torch.nn as nn
import torch.nn.functional as F

def gamma(x):
    return x.detach()

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
    def __init__(self, prototypes, proj_dim, similarity = 'cosine'):
        super().__init__()
        self.register_buffer('prototypes', prototypes)
        self.similarity = similarity
        self.projector = nn.Linear(prototypes.shape[1], proj_dim, bias=False)

    def forward(self, x):
        if self.similarity == 'cosine':
            x = F.normalize(x, dim=-1)
            p = F.normalize(self.projector(self.prototypes), dim=-1)
        else:
            p = self.projector(self.prototypes)
        sim = torch.einsum('bnd, kd->bnk', x, p)   #[b, n, k]
        sim_mean = sim.mean(dim=-1)         #[b, n]
        return sim, sim_mean
     
class HPMIL(nn.Module):
    def __init__(self, num_coarse, num_fine, coarse_proto, fine_proto, 
                 dropout=0., proj_dim = 512, embed_dim=1024):
        super(HPMIL, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attn = Attn_Net_Gated(L=proj_dim, D=256, n_classes=1)
        self.coarse_proto = PrototypeBlock(coarse_proto, proj_dim)
        self.fine_proto = PrototypeBlock(fine_proto, proj_dim)

        self.fc_coarse = nn.Linear(2*proj_dim, num_coarse)
        self.fc_fine = nn.Linear(proj_dim, num_fine)
    
    def forward(self, x):
        x = self.projector(x)
        # print("patch features shape: ", x.shape)
        attn_raw, _ = self.attn(x)
        attn_scores = F.softmax(attn_raw, dim=1)
        attn_scores = attn_scores.squeeze(-1)

        # print("attention scores shape: ", attn_scores.shape)
        _, sim_coarse = self.coarse_proto(x)
        _, sim_fine = self.fine_proto(x)

        sim_coarse = F.softmax(sim_coarse, dim=1)
        sim_fine = F.softmax(sim_fine, dim=1)

        # print("sim_coarse shape:", sim_coarse.shape)
        # print("sim_coarse shape:", sim_fine.shape)

        w_coarse = F.softmax(attn_scores*sim_coarse, dim=1)
        w_fine = F.softmax(attn_scores*sim_fine, dim=1)

        # print("w_coarse shape: ", w_coarse.shape)
        # print("w_fine shape: ", w_fine.shape)

        w_coarse = w_coarse.squeeze(-1)
        w_fine = w_fine.squeeze(-1)

        z_coarse = torch.einsum('bnd, bn->bd', x, w_coarse)
        z_fine = torch.einsum('bnd, bn->bd', x, w_fine)
        
        z_fine_grad_controller = gamma(z_fine)
        logits_coarse = self.fc_coarse(torch.cat([z_coarse, z_fine_grad_controller], dim=-1))  #[B, 4]

        logits_fine = self.fc_fine(z_fine)     #[B, 11]

        return logits_coarse, logits_fine
        

