"""
Inside ViT
Create three objects to implement ViT (vanilla)
1. Embedding
2.Transformer Encoder         (stack of encoders)
    MLP layer
    Normalization layer
    Multi-Head Attention layer
    Normalization layer (before multi-head attention)

3. MLP Head
"""
import torch
import torch.nn as nn

#create patches from an image
class CreatePatches(nn.Module):
    def __init__(self, patch_size):  #patch_size P
        super(CreatePatches, self).__init__()
        self.patch_size = patch_size

    def forward(self, img):            #similar to overloading () operator in c++ to use class like a function
        b, c, h, w = img.shape        #batch size, height, width, channels
        assert h % self.patch_size == 0 and w % self.patch_size == 0, "Image dimension must be divisible by patch size."
        num_patches = (h * w) // (self.patch_size * self.patch_size)
        patch_dim = c * self.patch_size * self.patch_size
        patches = img.unfold(2, self.patch_size, self.patch_size)       #horizontal slice across height
        patches = patches.unfold(3, self.patch_size, self.patch_size)       #vertical slice across width
        patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)   #flatten into 1D patches
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.reshape(b, num_patches, patch_dim)  #flatten each patch to a vector with elements C * P * P
        return patches

"""
add position embedding to the input patch embedding
input: patch embedding
output: position embedding

D denotes emb_dim
N denotes number of patches
"""
class CreateEmbedding(nn.Module):
    def __init__(self, patch_dim, emb_dim, num_patches):
        super(CreateEmbedding, self).__init__()
        self.proj = nn.Linear(patch_dim, emb_dim)           #linear projection of (PxPxC) x 1 to Dx1
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))     #create a learnable position embeddings, initialized with random numbers (1, N+1, D)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))       #create a learnable class token
    def forward(self, patches):
        b = patches.size(0)             #batch size
        x = self.proj(patches)              #projecting each patch to D x 1 from (P x P x C) x 1, in total D x N  (B, N, D)
        cls_tokens = self.cls_token.expand(b, -1, -1)           #duplicate class token across the batch, (B, 1, D)
        x = torch.cat((cls_tokens, x), dim = 1)         # (B, N+1, D)
        x = x + self.pos_embed                      #(B, N+1, D)
        return x

#contains two fc layers with a GELU non-linearity
class MLP(nn.Module):
    def __init__(self, emb_dim, mlp_dim, dropout = 0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(emb_dim, mlp_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.activation(self.fc1(x)))))   #dropout added

#a single encoder block
class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim, dropout = 0.2):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.multi_attention = nn.MultiheadAttention(emb_dim, num_heads, dropout = dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)      #for dropout
    def forward(self, x):
        attention_output, _ = self.multi_attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attention_output) #dropout in the residual
        x = x + self.mlp(self.norm2(x))
        return x

#for managing encoder pipeline  (encapsulation of the transformer encoder)
class Encoder(nn.Module):
    def __init__(self, depth, emb_dim, num_heads, mlp_dim, dropout = 0.2):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([EncoderBlock(emb_dim, num_heads, mlp_dim, dropout)
                                     for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class MLPHead(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super(MLPHead, self).__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        return self.fc(x[:, 0])

class ViTVanilla(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, emb_dim, depth, num_heads, mlp_dim, num_classes):
        super(ViTVanilla, self).__init__()
        num_patches = (img_size * img_size) // (patch_size * patch_size)
        patch_dim = in_channels * patch_size * patch_size

        self.patcher = CreatePatches(patch_size)
        self.embedder = CreateEmbedding(patch_dim, emb_dim, num_patches)
        self.encoder = Encoder(depth, emb_dim, num_heads, mlp_dim)
        self.head = MLPHead(emb_dim, num_classes)
    def forward(self, x):
        x = self.patcher(x)
        x = self.embedder(x)
        x = self.encoder(x)
        x = self.head(x)
        return x