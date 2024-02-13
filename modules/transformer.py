# References:
    # https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/transformer.py
    # https://nn.labml.ai/transformers/rope/index.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

torch.set_printoptions(linewidth=70)

# "Rotary encoding organizes the $d$ features as $d/2$ pairs. Each pair can be considered
# a coordinate in a 2D plane, and the encoding will rotate it by an angle depending on the position of the token."

# We pair feature $i$ with feature $i + 2/d$. So for position $m$
# If $i \in \{1, 2, ... d / 2\}$
# $$x^{(i)}_{m}$$
# is transformed to
# $$x^{(i)}_{m}\cos{m\theta_{i}} + (-x^{(i + d / 2)}_{m})\sin{m\theta_{i}}$$
# and otherwise transformed to
# $$x^{(i + d / 2)}_{m}\cos{m\theta_{i}} + x^{(i)}_{m}\sin{m\theta_{i}}$$
# $$\langle \text{RoPE}(x^{(1)}_{m}​, x^{(2)}_{m}​, m), \text{RoPE}(x^{(1)}_{n}​, x^{(2)}_{n}​, n) \rangle =\
# \langle \text{RoPE}(x^{(1)}_{m}​, x^{(2)}_{m}​, m - n), \text{RoPE}(x^{(1)}_{n}​, x^{(2)}_{n}​, 0) \rangle$$
# "This shows that for dot-production attention the rotary encodings gives relative attention."
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, base=10_000):
        super().__init__()

        self.head_dim = head_dim
        self.base = base

    def _get_theta(self, i):
        # $\Theta = \{\theta_{i} = 10000^{-2(i - 1)/d}, i \in [1, 2, \ldots d/2]\}$
        return self.base ** (-2 * (i - 1) / self.head_dim)

    # `x` is the tensor at the head of a key or a query with shape (`batch_size`, `n_heads`, `seq_len`, `head_dim`)
    def forward(self, x):
        _, _, seq_len, _ = x.shape

        pos = torch.arange(seq_len, dtype=x.dtype) # $m$
        i = torch.arange(1, self.head_dim // 2 + 1).repeat(2) # $i$ # 1, 2, ..., d / 2| 1, 2, ... d / 2
        theta = self._get_theta(i) # $\theta_{i}$
        v = torch.einsum("p,t->pt", pos, theta) # $m\theta_{i}$

        self.cos_mtheta = torch.cos(v) # $\cos{m\theta_{i}}$
        self.sin_mtheta = torch.sin(v) # $\sin{m\theta_{i}}$

        # 1, 2, ..., d // 2 - 1, d // 2, d // 2 + 1, ..., d - 1, d
        # -(1 + d / 2), -(2 + d / 2), ..., -(d - 1), -d, 1, ..., d / 2 - 1, d / 2
        pair = torch.cat([-x[..., self.head_dim // 2:], x[..., : self.head_dim // 2]], dim=3)
        x = x * self.cos_mtheta + pair * self.sin_mtheta
        return x


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim, pad_id):
        super().__init__(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_id,
        )


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim, pad_id=0):
        super().__init__()

        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pad_id = pad_id

        self.embed = nn.Embedding(num_embeddings=max_len + 1, embedding_dim=embed_dim, padding_idx=pad_id)

    def forward(self, x):
        not_pad = (x != self.pad_id)
        x = torch.cumsum(not_pad, dim=1) * not_pad
        x = self.embed(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, drop_prob):
        super().__init__()
    
        self.hidden_dim = hidden_dim # "$d_{model}$"
        self.n_heads = n_heads

        self.head_dim = hidden_dim // n_heads # "With a per-head state size of 64"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_drop = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @staticmethod
    def _get_attention_score(q, k):
        attn_score = torch.einsum("bnid,bnjd->bnij", q, k)
        return attn_score

    def forward(self, q, k, v, mask=None):
        b, i, _ = q.shape
        _, j, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(b, self.n_heads, i, self.head_dim)
        k = k.view(b, self.n_heads, j, self.head_dim)
        v = v.view(b, self.n_heads, j, self.head_dim)

        attn_score = self._get_attention_score(q=q, k=k)
        if mask is not None:
            attn_score.masked_fill_(mask=mask, value=-1e9)
        attn_score /= (self.head_dim ** 0.5)
        attn_weight = F.softmax(attn_score, dim=3)

        attn_weight_drop = self.attn_drop(attn_weight)
        x = torch.einsum("bnij,bnjd->bnid", attn_weight_drop, v)
        x = rearrange(x, pattern="b n i d -> b i (n d)")

        x = self.out_proj(x)
        return x, attn_weight


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, drop_prob, activ="relu"):
        super().__init__()

        assert activ in ["relu", "gelu"], (
            """The argument `activ` must be one of (`"relu"`, `"gelu"`)"""
        )

        self.activ = activ

        self.proj1 = nn.Linear(hidden_dim, mlp_dim)
        if activ == "relu":
            self.relu = nn.ReLU()
        else:
            self.gelu = nn.GELU()
        self.proj2 = nn.Linear(mlp_dim, hidden_dim)
        self.mlp_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj1(x)
        if self.activ == "relu":
            x = self.relu(x)
        else:
            x = self.gelu(x)
        x = self.proj2(x)
        x = self.mlp_drop(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, hidden_dim, drop_prob):
        super().__init__()

        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.resid_drop = nn.Dropout(drop_prob)

    def forward(self, x, sub_layer):
        skip = x.clone()
        x = self.pre_norm(x)
        x = sub_layer(x)
        x = self.resid_drop(x)
        return x + skip


class DecoderLayer(nn.Module):
    def __init__(
        self, n_heads, hidden_dim, mlp_dim, attn_drop_prob, ff_drop_prob, resid_drop_prob,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        self.self_attn = MultiHeadAttention(hidden_dim=hidden_dim, n_heads=n_heads, drop_prob=attn_drop_prob)
        self.self_attn_resid_conn = ResidualConnection(hidden_dim=hidden_dim, drop_prob=resid_drop_prob)
        self.feed_forward = PositionwiseFeedForward(
            hidden_dim=hidden_dim, mlp_dim=mlp_dim, drop_prob=ff_drop_prob, activ="relu",
        )
        self.ff_resid_conn = ResidualConnection(hidden_dim=hidden_dim, drop_prob=resid_drop_prob)

    def forward(self, x, self_attn_mask):
        x = self.self_attn_resid_conn(
            x, sub_layer=lambda x: self.self_attn(q=x, k=x, v=x, mask=self_attn_mask)[0],
        )
        x = self.ff_resid_conn(x, sub_layer=self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        pad_id,
        n_heads,
        hidden_dim,
        mlp_dim,
        n_layers,
        embed_drop_prob,
        attn_drop_prob,
        ff_drop_prob,
        resid_drop_prob,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, embed_dim=hidden_dim, pad_id=pad_id)
        self.pos_embed = LearnedPositionalEmbedding(max_len=max_len, embed_dim=hidden_dim, pad_id=pad_id)
        rope = RotaryPositionalEmbedding(dim=hidden_dim)
        self.embed_drop = nn.Dropout(embed_drop_prob)

        self.dec_stack = nn.ModuleList(
            [
                DecoderLayer(
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    attn_drop_prob=attn_drop_prob,
                    ff_drop_prob=ff_drop_prob,
                    resid_drop_prob=resid_drop_prob,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, self_attn_mask=None):
        x = self.token_embed(x) + self.pos_embed(x)
        x = self.embed_drop(x)
        for dec_layer in self.dec_stack:
            x = dec_layer(x, self_attn_mask=self_attn_mask)
        x = self.linear(x)
        return x


DROP_PROB = 0.1
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        pad_id,
        hidden_dim,
        mlp_dim,
        n_layers,
        n_heads,
        embed_drop_prob=DROP_PROB,
        attn_drop_prob=DROP_PROB,
        ff_drop_prob=DROP_PROB,
        resid_drop_prob=DROP_PROB,
    ):
        super().__init__()

        self.max_len = max_len
        self.pad_id = pad_id

        self.dec = Decoder(
            vocab_size=vocab_size,
            max_len=max_len,
            pad_id=pad_id,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            n_layers=n_layers,
            embed_drop_prob=embed_drop_prob,
            attn_drop_prob=attn_drop_prob,
            ff_drop_prob=ff_drop_prob,
            resid_drop_prob=resid_drop_prob,
        )

        # self.dec.linear.weight = self.dec.input.embed.weight

    def _get_pad_mask(self, x):
        mask = (x == self.pad_id).unsqueeze(1).unsqueeze(2)
        return mask

    def _get_causal_mask(self):
        ones = torch.ones(size=(self.max_len, self.max_len))
        mask = torch.triu(ones, diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(1)
        return mask

    def forward(self, x):
        pad_mask = self._get_pad_mask(x)
        causal_mask = self._get_causal_mask()
        return self.dec(x, self_attn_mask=pad_mask | causal_mask)


if __name__ == "__main__":
    BATCH_SIZE = 1
    TEXT_VOCAB_SIZE = 16384
    IMG_VOCAB_SIZE = 8192
    TEXT_MAX_LEN = 256
    IMG_TOKEN_SIZE = 32
    IMG_MAX_LEN = IMG_TOKEN_SIZE ** 2
    HIDDEN_DIM = 3968 # "$d_{\text{model}}$"
    # HIDDEN_DIM = 128
    PAD_ID = 0
    # "We use 64 attention layers, each of which uses 62 attention heads."
    # N_LAYERS = 64
    N_LAYERS = 2
    N_HEADS = 62

    seq = torch.randint(
        low=0,
        high=IMG_VOCAB_SIZE,
        size=(BATCH_SIZE, TEXT_MAX_LEN + IMG_MAX_LEN),
    )
    seq[:, -2:] = PAD_ID

    model = Transformer(
        vocab_size=IMG_VOCAB_SIZE,
        max_len=TEXT_MAX_LEN + IMG_MAX_LEN,
        pad_id=PAD_ID,
        hidden_dim=HIDDEN_DIM,
        mlp_dim=HIDDEN_DIM * 4,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
    )
    out = model(seq)
