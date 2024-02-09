import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

torch.set_printoptions(linewidth=70)


# "The use of 1 1 convolutions at the end of the encoder and the beginning of the decoder."
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


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeds, hidden_dim):
        super().__init__()

        self.embed_space = nn.Embedding(n_embeds, hidden_dim)
        self.embed_space.weight.data.uniform_(-1 / n_embeds, 1 / n_embeds) # Uniform distribution??

    def vector_quantize(self, x): # (b, `n_embeds`, h, w)
        b, _, h, w = x.shape
        x = rearrange(x, pattern="b c h w -> (b h w) c")
        squared_dist = ((x.unsqueeze(1) - self.embed_space.weight.unsqueeze(0)) ** 2).sum(dim=2)
        # using the shared embedding space $e$.
        argmin = torch.argmin(squared_dist, dim=1) # (b * h * w,)
        q = argmin.view(b, h, w) # (b, h, w)
        return q

    def forward(self, x):
        q = self.vector_quantize(x) # (b, h, w)
        x = self.embed_space(q) # (b, h, w, `hidden_dim`)
        return x.permute(0, 3, 1, 2)


class Decoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()

        # The decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions
        # with stride 2 and window size 4 × 4.
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
    def __init__(self, channels, n_embeds, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.enc = Encoder(channels=channels, hidden_dim=hidden_dim)
        self.vect_quant = VectorQuantizer(n_embeds=n_embeds, hidden_dim=hidden_dim)
        self.dec = Decoder(channels=channels, hidden_dim=hidden_dim)

    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, z):
        x = self.dec(z)
        return x

    def forward(self, ori_image):
        z_e = self.encode(ori_image)
        z_q = self.vect_quant(z_e)
        x = self.decode(z_q)
        return x

    def get_vqvae_loss(self, ori_image, commit_weight=0.25):
        z_e = self.encode(ori_image)
        z_q = self.vect_quant(z_e)
        # This term trains the vector quantizer.
        vq_loss = F.mse_loss(z_e.detach(), z_q, reduction="mean")
        # This term trains the encoder.
        commit_loss = commit_weight * F.mse_loss(z_e, z_q.detach(), reduction="mean")
        z_q = z_e + (z_q - z_e).detach() # Preserve gradient??

        recon_image = self.decode(z_q)
        # This term trains the decoder.
        recon_loss = F.mse_loss(recon_image, ori_image, reduction="mean")
        return recon_loss + vq_loss + commit_loss

    def get_prior_q(self, ori_image):
        z_e = self.encode(ori_image)
        q = self.vect_quant.vector_quantize(z_e)
        return q

    def get_pixelcnn_loss(self, ori_image):
        with torch.no_grad():
            q = self.get_prior_q(ori_image)
        pred_q = self.pixelcnn(q.detach())
        return self.ce(
            rearrange(pred_q, pattern="b c h w -> (b h w) c"), q.view(-1,),
        )

    @staticmethod
    def deterministically_sample(x):
        return torch.argmax(x, dim=1)

    @staticmethod
    def stochastically_sample(x, temp=1):
        b, c, h, w = x.shape
        prob = F.softmax(x / temp, dim=1)
        sample = torch.multinomial(prob.view(-1, c), num_samples=1, replacement=True)
        return sample.view(b, h, w)

    def q_to_image(self, q):
        x = self.vect_quant.embed_space(q)
        return self.decode(x.permute(0, 3, 1, 2))

    def sample(self, batch_size, q_size, device, temp=0):
        post_q = self.sample_post_q(
            batch_size=batch_size, q_size=q_size, device=device, temp=temp,
        )
        return self.q_to_image(post_q)


if __name__ == "__main__":
    IMG_SIZE = 256
    IMG_TOKEN_SIZE = 32
    CHANNELS = 3
    IMG_VOCAB_SIZE = 8192
    HIDDEN_DIM = 256
    model = dVAE(channels=CHANNELS, n_embeds=IMG_VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    x = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE)
    out = model(x)
