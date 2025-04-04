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
import torch.nn.functional as F
import torch.nn as nn

#create patches from an image
class createPatches(nn.Module):
    def __call__(self, img):            #similar to overloading () operator in c++ to use class as a function
        img.shape = h, w, c     #height, width, channels
        #patches =
"""
linear projection of flattened 2D patches
"""
class linProjectPatches(nn.Module):
    def __init__(self, input):
        self.input = input
    def forward(self, x):
        pass
"""
add position embedding to the input patch embedding
input: patch embedding
output: position embedding
"""
class CreateEmbedding(nn.Module):
    def __init__(self):
        super(CreateEmbedding, self).__init__()
    def __call__(self, *args, **kwargs):
        pass

#contains two layers with a GELU non-linearity
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

#a single encoder block
class Encoder1D(nn.Module):
    def __call__(self, embPatch):
        x = nn.LayerNorm(embPatch)              #Normalize
        #x = nn.MultiheadAttention()
        x = x + embPatch                    #residual connection to prevent vanishing gradient
        prev = x
        x = nn.LayerNorm(x)
        x = MLP(x)
        x =  x + prev
        return x

#for managing encoder pipeline  (encapsulation of the transformer encoder)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
    def __call__(self, *args, **kwargs):
        pass


class ViTVanilla():
    """
    pseudocode

    transformer_input = CreateEmbedding() class that combines patch embedding with positional embedding
    encoder_output = Encoder(transformer_input)
    prediction = MLP(encoder_output)
    return prediction
    """
    pass

