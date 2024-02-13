# References:
    # https://github.com/mszulc913/dvae-pytorch/blob/main/dvae_pytorch/models/dvae.py
    # https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/vae.py

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

torch.set_printoptions(linewidth=70)


# "Increasing the KL weight to $\beta = 6.6$ promotes better codebook usage and ultimately leads to a Zero-Shot Text-to-Image Generation smaller reconstruction error at the end of training."
# "The use of 1 1 convolutions at the end of the encoder and the beginning of the decoder."
# "Multiplication of the outgoing activations from the encoder and decoder resblocks by a small constant, to ensure stable training at initialization."
# "The models primarily use 3   3 convolutions, with 1   1 convolutions along skip connections in which the number of feature maps changes between the input and output of a resblock. The first convolution of the encoder is 7   7, and the last convolution of the encoder (which produces the 32   32   8192 output used as the logits for the categorical distributions for the image tokens) is 1   1. Both the first and last convolutions of the decoder are 1   1. The encoder uses max-pooling (which we found to yield better ELB than average-pooling) to downsample the feature maps, and the decoder uses nearest-neighbor upsampling."

class Encoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        x = self.conv_block(x)
        x = x + self.res_block(x)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()

        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, channels, 4, 2, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        x = self.conv_block(x)
        return x


class dVAE(nn.Module):
    def __init__(self, channels, codebook_size, hidden_dim):
        super().__init__()

        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim

        self.enc = Encoder(channels=channels, hidden_dim=hidden_dim)
        self.codebook = nn.Embedding(codebook_size, hidden_dim)
        # self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)
        self.dec = Decoder(channels=channels, hidden_dim=hidden_dim)


    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, z):
        x = self.dec(z)
        return x

    def forward(self, ori_image, temp, hard):
        x = self.encode(ori_image)
        print(x.shape, self.codebook.weight.data.shape)
        x = F.gumbel_softmax(x, tau=temp, hard=hard, dim=1)
        x = torch.einsum("bchw,dc->bdhw", x, self.codebook.weight.data)
        return self.decode(x)

    def get_loss(self, ori_image, temp, hard, kl_weight):
        x = self.encode(ori_image) # (b, `hidden_dim`, h, w)

        softmax = F.softmax(x.permute((0, 2, 3, 1)), dim=1)
        uniform = torch.ones_like(softmax) / self.codebook.size
        elbo_loss = F.kl_div(softmax, uniform, log_target=False, reduction="batchmean")

        x = F.gumbel_softmax(x, tau=temp, hard=hard, dim=1)
        x = torch.einsum("bchw,cd->bdhw", x, self.codebook.weight.data)
        recon_image = self.decode(x)
        recon_loss = F.mse_loss(recon_image, ori_image, reduction="mean")
        return recon_loss + kl_weight * elbo_loss


if __name__ == "__main__":
    IMG_SIZE = 256
    IMG_TOKEN_SIZE = 32
    CHANNELS = 3
    IMG_VOCAB_SIZE = 8192
    HIDDEN_DIM = 256
    TEMP = 1
    KL_WEIGHT = 3
    model = dVAE(channels=CHANNELS, codebook_size=IMG_VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    x = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
    out = model(x, temp=TEMP, hard=False)
