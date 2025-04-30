import torch
from torch import nn
from torch.functional import F
from models.model_CLAM import CLAM_SB
class Aggregator(nn.Module):            #CLAM wrapper
    def __init__(self, gate = True, size_arg = "small", dropout = 0.25, k_sample = 8, n_classes = 2, subtyping = True, embed_dim = 1024):
        super().__init__()
        self.gate = gate
        self.size_arg = size_arg
        self.dropout = dropout
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.embed_dim = embed_dim
    def forward(self, feature_vector):
        aggregator = CLAM_SB(self.gate, self.size_arg, self.dropout, self.k_sample, self.n_classes, self.subtyping, self.embed_dim)
        return aggregator(feature_vector)


class Classifier(nn.Module):
    def __init__(self, levels = 2, nc = 4, nf = 14, dropout = 0.25, input_dim = 512):      #default values chosen based on our experiment setting
        super().__init__()
        assert input_dim % levels == 0, "Input dimension must be divisible by k"
        self.input_dim = input_dim
        self.levels = levels          #hierarchy levels
        self.nc = nc
        self.nf = nf
        self.dropout = dropout
        self.feature_dim = int(input_dim / self.levels)

        projection_fc = [nn.Linear(self.input_dim, self.feature_dim), nn.ReLU(), nn.Dropout(self.dropout)]
        self.projection_fc = nn.Sequential(*projection_fc)
        projection_head_c = [self.fc for _ in range(self.nc)]
        projection_head_f = [self.fc for _ in range(self.nf)]
        
        self.projection_head_c = nn.ModuleList(projection_head_c)
        self.projection_head_f = nn.ModuleList(projection_head_f)

        classification_fc = [nn.Linear(self.feature_dim, 1), nn.ReLU(), nn.Dropout(self.dropout)]
        self.classification_fc = nn.Sequential(*classification_fc)
        classification_heads = [self.classification_fc for _ in range(self.levels)]
        self.classification_heads = nn.ModuleList(classification_heads)

        # self.classification_head_c = nn.Linear(self.feature_dim, 1)
        # self.classification_head_f = nn.Linear(self.feature_dim, 1)

    def forward(self, slide_feature):
        partitions = torch.split(slide_feature, self.levels)
        coarse = partitions[0]              #[1, 256]
        fine = partitions[1]                   #[1,256]

        coarse_input = torch.cat([coarse, fine.detach()])  #[1, 512]
        fine_input = torch.cat([fine, coarse.detach()])     #[1, 512]

        coarse_features = torch.stack([clf(coarse_input) for clf in self.projection_head_c])       # [nc, 256]
        fine_features = torch.stack([clf(fine_input) for clf in self.projection_head_f])            # [nf, 256]

        coarse_logits = self.classification_heads[0](coarse_features)       # [nc, 1]
        prob_coarse = F.softmax(coarse_logits, dim = 0)                 # [nc, 1]
        pred_coarse_idx = torch.topk(prob_coarse, k = 1, dim = 0)[1]        # [1]

        fine_logits = self.classification_heads[1](fine_features)       # [nf, 1]
        prob_fine = F.softmax(fine_logits, dim = 0)                     # [nf, 1]
        pred_fine_idx = torch.topk(prob_fine, k = 1, dim = 0)[1]            #[1]

        pred_coarse = torch.index_select(prob_coarse, dim = 0, index = pred_coarse_idx)     # [1]
        pred_fine = torch.index_select(prob_fine, dim = 0, index = pred_fine_idx)           # [1]

        return pred_coarse, pred_fine


class HiClass(nn.Module):
    def __init__(self, nc, nf, levels):
        super().__init__()
        self.aggregator = Aggregator()
        self.classifier = Classifier(levels = levels, nc = nc, nf = nf)
        # self.coarse_labels = {0 : "Benign", 1 : "Cancer", 2 : "Dysplasia", 3 : "Gastritis"}
        # self.fine_labels = {0 : "fundic gland polyp", 1 : "hyperplastic polyp." , 2 : "xanthoma", 3 : "adenocarcinoma",
        #                     4 : "tubular adenocarcinoma", 5 : "poorly cohesive carcinoma", 6 : "low grade dysplasia",
        #                     7 : "high grade dysplasia", 8 : "chronic active gastritis", 9 : "chronic gastritis",
        #                     10 : "erosion", 11 : "intestinal metaplasia", 12 : "lymphoid aggregate", 13: "ulceration"}

    def forward(self, x):
        x = self.aggregator(x)
        coarse_class, fine_class = self.classifier(x)
        return coarse_class, fine_class