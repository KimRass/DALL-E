# References:
    # https://github.com/mszulc913/dvae-pytorch/blob/main/dvae_pytorch/training/lightning.py
    # https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/vae.py
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/cat_vae.py
    # https://homes.cs.washington.edu/~ewein//blog/2022/03/04/gumbel-max/
    # https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
    # https://github.com/hugobb/discreteVAE/blob/master/train_gumbel_vae.py

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

torch.set_printoptions(linewidth=70)


# "Increasing the KL weight to $\beta = 6.6$ promotes better codebook usage and ultimately leads to a Zero-Shot Text-to-Image Generation smaller reconstruction error at the end of training."
# "The use of 1 1 convolutions at the end of the encoder and the beginning of the decoder."
# "Multiplication of the outgoing activations from the encoder and decoder resblocks by a small constant, to ensure stable training at initialization."
# "The models primarily use 3   3 convolutions, with 1   1 convolutions along skip connections in which the number of feature maps changes between the input and output of a resblock. The first convolution of the encoder is 7   7, and the last convolution of the encoder (which produces the 32   32   8192 output used as the logits for the categorical distributions for the image tokens) is 1   1. Both the first and last convolutions of the decoder are 1   1. The encoder uses max-pooling (which we found to yield better ELB than average-pooling) to downsample the feature maps, and the decoder uses nearest-neighbor upsampling."


class ConvBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
        )
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x + self.res_block(x)
        return self.downsample(x)


class Encoder(nn.Module):
    def __init__(self, hidden_dim, codebook_size, channels=3):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 7, 1, 3), # First
            ConvBlock(hidden_dim),
            ConvBlock(hidden_dim * 2),
            ConvBlock(hidden_dim * 4),
            nn.Conv2d(hidden_dim * 8, codebook_size, 1, 1, 0) # Last
        )

    def forward(self, x):
        return self.layers(x)


class dVAE(nn.Module):
    def __init__(self, channels, codebook_size, codebook_dim, hidden_dim):
        super().__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_dim = hidden_dim

        self.enc = Encoder(channels=channels, hidden_dim=hidden_dim)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        # self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)
        # self.dec = Decoder(channels=channels, hidden_dim=hidden_dim)

    def encode(self, x, temp, hard):
        enc_out = self.enc(x) # (b, `codebook_size`, h, w)
        print(enc_out.shape)
        x = F.gumbel_softmax(enc_out.permute((0, 2, 3, 1)), tau=temp, hard=hard, dim=1)
        x = torch.einsum("bchw,dc->bdhw", x, self.codebook.weight.data) # (b, `codebook_dim`, h, w)
        return x, enc_out.permute((0, 2, 3, 1))

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
    CHANNELS = 3
    IMG_SIZE = 256
    HIDDEN_DIM = 32
    IMG_VOCAB_SIZE = 8192
    TEMP = 1
    enc = Encoder(hidden_dim=HIDDEN_DIM, codebook_size=IMG_VOCAB_SIZE)
    x = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
    enc_out = enc(x)
    hard_F.gumbel_softmax(enc_out, tau=TEMP, hard=True, dim=1)



    
    IMG_TOKEN_SIZE = 32
    IMG_VOCAB_DIM = 32
    TEMP = 1
    KL_WEIGHT = 3
    enc(torch.randn(1, CHANNELS, 256, 256)).shape
    enc(torch.randn(1, CHANNELS, 128, 128)).shape
    
    model = dVAE(channels=CHANNELS, codebook_size=IMG_VOCAB_SIZE, hidden_dim=HIDDEN_DIM, codebook_dim=IMG_VOCAB_DIM)
    model.encode(x, 1, True)
    out = model(x, temp=TEMP, hard=False)


# latent_dim = 32
# categorical_dim = 64
# temperature = 1
# def sample_gumbel(shape, eps=1e-20):
#     U = torch.rand(shape)
#     return -torch.log(-torch.log(U + eps) + eps)
# def gumbel_softmax_sample(logits, temperature):
#     y = logits + sample_gumbel(logits.size())
#     return F.softmax(y / temperature, dim=-1)
# def gumbel_softmax(logits, temperature, hard=False):
#     """
#     ST-gumple-softmax
#     input: [*, n_class]
#     return: flatten --> [*, n_class] an one-hot vector
#     """
#     y = gumbel_softmax_sample(logits, temperature)

#     # hard = True
#     if not hard:
#         return y.view(-1, latent_dim * categorical_dim)

#     shape = y.size()
#     _, ind = y.max(dim=-1)
#     y_hard = torch.zeros_like(y).view(-1, shape[-1])
#     y_hard.scatter_(1, ind.view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
#     # Set gradients w.r.t. y_hard gradients w.r.t. y
#     y_hard = (y_hard - y).detach() + y
#     return y_hard.view(-1, latent_dim * categorical_dim)
#     # return y_hard

# logits = torch.randn(4, latent_dim, categorical_dim)
# gumbel_softmax(logits, 1, False).shape
# gumbel_softmax(logits, 1, True).shape
# gumbel_softmax_sample(logits, 1)

