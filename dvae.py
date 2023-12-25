from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange
# from dalle_pytorch import distributed_utils

# from axial_positional_embedding import AxialPositionalEmbedding
# from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE
# from dalle_pytorch.transformer import Transformer, DivideMax


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# class always():
#     def __init__(self, val):
#         self.val = val
#     def __call__(self, x, *args, **kwargs):
#         return self.val


# def is_empty(t):
#     return t.nelement() == 0


# def masked_mean(t, mask, dim = 1):
#     t = t.masked_fill(~mask[:, :, None], 0.)
#     return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]


# def prob_mask_like(shape, prob, device):
#     return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


# def set_requires_grad(model, value):
#     for param in model.parameters():
#         param.requires_grad = value


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


# def gumbel_noise(t):
#     noise = torch.zeros_like(t).uniform_(0, 1)
#     return -log(-log(noise))


# def gumbel_sample(t, temperature = 1., dim = -1):
#     return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)


# def top_k(logits, thres = 0.5):
#     num_logits = logits.shape[-1]
#     k = max(int((1 - thres) * num_logits), 1)
#     val, ind = torch.topk(logits, k)
#     probs = torch.full_like(logits, float('-inf'))
#     probs.scatter_(1, ind, val)
#     return probs


# class SharedEmbedding(nn.Embedding):
#     def __init__(self, linear, start_index, end_index, **kwargs):
#         super().__init__(end_index - start_index, linear.weight.shape[1], **kwargs)
#         del self.weight

#         self.linear = linear
#         self.start_index = start_index
#         self.end_index = end_index

#     def forward(self, input):
#         return F.embedding(
#             input, self.linear.weight[self.start_index:self.end_index], self.padding_idx, self.max_norm,
#             self.norm_type, self.scale_grad_by_freq, self.sparse)


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
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
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.channels = channels
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.reinmax = reinmax

        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        # hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = list()
        dec_layers = list()

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = tuple(map(lambda t: t[:channels], normalization))

    #     self._register_external_parameters()

    # def _register_external_parameters(self):
    #     """Register external parameters for DeepSpeed partitioning."""
    #     if (
    #             not distributed_utils.is_distributed
    #             or not distributed_utils.using_backend(
    #                 distributed_utils.DeepSpeedBackend)
    #     ):
    #         return

    #     deepspeed = distributed_utils.backend.backend_module
    #     deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'

        img = self.norm(img)

        logits = self.encoder(img)

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)

        one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)

        if self.straight_through and self.reinmax:
            # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
            # algorithm 2
            one_hot = one_hot.detach()
            π0 = logits.softmax(dim = 1)
            π1 = (one_hot + (logits / temp).softmax(dim = 1)) / 2
            π1 = ((log(π1) - logits).detach() + logits).softmax(dim = 1)
            π2 = 2 * π1 - 0.5 * π0
            one_hot = π2 - π2.detach() + one_hot

        sampled = einsum('b n h w, n d -> b d h w', one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim = -1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss
        return loss, out

if __name__ == "__main__":
    vae = DiscreteVAE(
        image_size = 256,
        num_layers = 3,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
        num_tokens = 8192,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
        codebook_dim = 512,       # codebook dimension
        hidden_dim = 64,          # hidden dimension
        num_resnet_blocks = 1,    # number of resnet blocks
        temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
        straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
    )
    image = torch.randn(4, 3, 256, 256)
    out = vae(image, return_loss=False)
    print(out.shape)

    # loss = vae(image, return_loss=True)
    # loss
