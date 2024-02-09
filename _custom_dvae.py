# References:
    # https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        n_tokens = 512, # "$K$"
        codebook_dim = 512, # "$D$"
        n_layers = 3,
        n_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        reinmax = False,
        kl_div_loss_weight = 0.,
        normalization = ((*((0.5,) * 3), 0), (*((0.5,) * 3), 1))
    ):
        super().__init__()


n_tokens = 512
codebook_dim = 512
n_layers = 3
hidden_dim = 64
channels = 3
n_resnet_blocks = 0

has_resnet_block = n_resnet_blocks > 0

codebook = nn.Embedding(n_tokens, codebook_dim) # "$e = R^{K \times D}$"

enc_chans = [hidden_dim] * n_layers
dec_chans = list(reversed(enc_chans))

enc_chans = [channels, *enc_chans]
enc_chans

dec_init_chan = codebook_dim if not has_resnet_block else dec_chans[0]
dec_chans = [dec_init_chan, *dec_chans]
dec_chans
