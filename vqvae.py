import torch
from torch import nn
from torch.nn import functional as F

# "We train these models using the same standard VAE architecture on CIFAR10, while varying the latent capacity (number of continuous or discrete latent variables, as well as the dimensionality of the discrete space K)."


class ResidualBlock(nn.Module):
    def __init__(self, channels1, channels2):
        super().__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=channels1,
            out_channels=channels2,
            kernel_size=3,
            padding=1,
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channels2,
            out_channels=channels1,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


# "Implemented as ReLU, 3 × 3 conv, ReLU, 1 × 1 conv"
class ResidualStack(nn.Module):
    def __init__(self, n_blocks, channels1, channels2):
        super().__init__()

        self.resid_blocks = nn.ModuleList(
            [
                ResidualBlock(channels1=channels1, channels2=channels2)
                for _ in range(n_blocks)
            ]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        for block in self.resid_blocks:
            x += block(x)
        x = self.relu(x)
        return x


# "The encoder consists of 2 strided convolutional layers with stride 2 and window size 4 × 4, followed by two residual 3 × 3 blocks, all having 256 hidden units."
class Encoder(nn.Module):
    def __init__(
        self,
        n_downsampling_layers=2,
        channels=256,
        n_resid_blocks=2,
        resid_channels=32,
    ):
        super().__init__()

        # The last ReLU from the Sonnet example is omitted because ResidualStack starts
        # off with a ReLU.
        self.layers = nn.Sequential()
        for idx in range(1, n_downsampling_layers + 1):
            if idx == 0:
                in_channels=3
                out_channels = channels // 2
            elif idx == 1:
                in_channels = channels // 2
                out_channels = channels
            else:
                in_channels = channels
                out_channels = channels

            self.layers.add_module(
                name=f"down{idx}",
                module=nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            self.layers.add_module(name=f"relu{idx}", module=nn.ReLU())

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.resid_stack = ResidualStack(
            n_blocks=n_resid_blocks,
            channels1=channels,
            channels2=resid_channels,
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.conv(x)
        return self.resid_stack(x)


# "The decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions with stride 2 and window size 4 × 4."
class Decoder(nn.Module):
    def __init__(
        self,
        n_upsampling_layers=2,
        channels=256,
        embedding_dim=64,
        n_resid_blocks=2,
        resid_channels=32,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=embedding_dim,
            # in_channels=resid_channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.resid_stack = ResidualStack(
            n_blocks=n_resid_blocks,
            channels1=channels,
            channels2=resid_channels,
        )
        self.layers = nn.Sequential()
        for idx in range(1, n_upsampling_layers + 1):
            if idx < n_upsampling_layers - 2:
                in_channels = channels
                out_channels = channels
            elif idx == n_upsampling_layers - 2:
                in_channels = channels
                out_channels = channels // 2
            else:
                in_channels = channels // 2
                out_channels = 3

            self.layers.add_module(
                name=f"up{idx}",
                module=nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            if idx < n_upsampling_layers - 1:
                self.layers.add_module(name=f"relu{idx}", module=nn.ReLU())


    def forward(self, x):
        x = self.conv(x)
        x = self.resid_stack(x)
        x = self.layers(x)
        return x


class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, n_embeddings, use_ema, decay, epsilon):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon

        # Dictionary embeddings.
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, n_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (n_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)

        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.n_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.N_i_ts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                # This is kind of weird.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.n_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        n_downsampling_layers,
        n_resid_blocks,
        resid_channels,
        embedding_dim,
        n_embeddings,
        use_ema,
        decay,
        epsilon,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            channels,
            n_downsampling_layers,
            n_resid_blocks,
            resid_channels,
        )
        self.pre_vq_conv = nn.Conv2d(
            in_channels=channels, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantizer(
            embedding_dim, n_embeddings, use_ema, decay, epsilon
        )
        self.decoder = Decoder(
            embedding_dim,
            channels,
            n_downsampling_layers,
            n_resid_blocks,
            resid_channels,
        )

    def quantize(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        (
            z_quantized,
            dictionary_loss,
            commitment_loss,
            encoding_indices,
        ) = self.vq(z)
        return (
            z_quantized,
            dictionary_loss,
            commitment_loss,
            encoding_indices,
        )

    def forward(self, x):
        (z_quantized, dictionary_loss, commitment_loss, _) = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": x_recon,
        }

if __name__ == "__main__":
    # channels = 128
    channels = 256
    n_downsampling_layers = 2
    n_resid_blocks = 2
    resid_channels = 32
    n_upsampling_layers = 2
    embedding_dim = 64
    dec = Decoder(
        n_upsampling_layers=n_upsampling_layers,
        channels=channels,
        embedding_dim=embedding_dim,
        n_resid_blocks=n_resid_blocks,
        resid_channels=resid_channels,
    )
