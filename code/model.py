"""SBi-Transformer (Sections 3.2-3.4 of the paper) + switches for Table 4/5 ablations."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMHA(nn.Module):
    """Global multi-head self-attention — Eq. (5), (6)."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dk).transpose(1, 2)

    def forward(self, x):
        q, k, v = self._split(self.wq(x)), self._split(self.wk(x)), self._split(self.wv(x))
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        a = self.drop(torch.softmax(s, -1))
        o = (a @ v).transpose(1, 2).reshape(x.shape)
        return self.wo(o)


class SparseMHA(nn.Module):
    """Local sparse self-attention — Eq. (7), (8).

    Binary mask M = (local window) OR (mutual nearest neighbors).
    The paper only states "local window + mutual nearest neighbor selection" and
    `block_size = 16`; our interpretation: window half-width = block_size//2,
    and k of the mutual-kNN = block_size//2 (see README, "Assumptions" section).
    """

    def __init__(self, d_model, n_heads, dropout, block_size=16):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.block = block_size
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dk).transpose(1, 2)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self._split(self.wq(x)), self._split(self.wk(x)), self._split(self.wv(x))
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dk)

        w = max(1, self.block // 2)
        idx = torch.arange(T, device=x.device)
        local = (idx[None, :] - idx[:, None]).abs() <= w          # (T,T)

        kk = min(max(1, self.block // 2), T)
        top = torch.zeros_like(s, dtype=torch.bool)
        top.scatter_(-1, s.topk(kk, dim=-1).indices, True)
        mutual = top & top.transpose(-2, -1)                       # mutual neighbors

        M = mutual | local
        s = s.masked_fill(~M, float("-inf"))
        a = self.drop(torch.softmax(s, -1))
        o = (a @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(o)


class EncoderLayer(nn.Module):
    """Encoder layer: two-tier attention -> FFN -> residual + LayerNorm (Eq. 9)."""

    def __init__(self, d_model, n_heads, ffn_hidden, dropout, block_size,
                 use_global=True, use_sparse=True):
        super().__init__()
        self.use_global, self.use_sparse = use_global, use_sparse
        if use_global:
            self.mha = GlobalMHA(d_model, n_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
        if use_sparse:
            self.sparse = SparseMHA(d_model, n_heads, dropout, block_size)
            self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_hidden), nn.ReLU(),
                                 nn.Linear(ffn_hidden, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_global:
            x = self.norm1(x + self.drop(self.mha(x)))
        if self.use_sparse:
            x = self.norm2(x + self.drop(self.sparse(x)))
        return self.norm3(x + self.drop(self.ffn(x)))


class SBiTransformer(nn.Module):
    def __init__(self, input_size=17, seq_len=45, num_hidden=16, ffn_hidden=32,
                 n_heads=2, encoder_layers=3, dropout=0.2, bilstm_size=32,
                 num_layers=2, block_size=16,
                 use_transformer=True, use_sparse=True, use_bilstm=True):
        super().__init__()
        self.use_transformer, self.use_sparse = use_transformer, use_sparse
        self.use_bilstm = use_bilstm
        # Eq. (4): linear projection + LEARNED position embedding
        self.proj = nn.Linear(input_size, num_hidden)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, num_hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        if use_transformer:
            for _ in range(encoder_layers):
                self.layers.append(EncoderLayer(num_hidden, n_heads, ffn_hidden,
                                                dropout, block_size,
                                                use_global=True, use_sparse=use_sparse))
        if use_bilstm:
            self.bilstm = nn.LSTM(num_hidden, bilstm_size, num_layers=num_layers,
                                  batch_first=True, bidirectional=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
            head_in = 2 * bilstm_size
        else:
            head_in = num_hidden
        self.head = nn.Linear(head_in, 1)                       # Eq. (12)

    def forward(self, x):
        h = self.drop(self.proj(x) + self.pos)
        for l in self.layers:
            h = l(h)
        if self.use_bilstm:
            h, _ = self.bilstm(h)                               # Eq. (10), (11)
        return self.head(h[:, -1]).squeeze(-1)


ABLATIONS = {
    # name -> (Transformer, Multi-head Sparse Attention, BiLSTM)  — order as in Table 4/5
    "no_no_yes":  (False, False, True),
    "yes_no_no":  (True,  False, False),
    "yes_no_yes": (True,  False, True),
    "yes_yes_no": (True,  True,  False),
    "yes_yes_yes": (True, True,  True),
}


def build_model(cfg, ablation="yes_yes_yes"):
    ut, us, ub = ABLATIONS[ablation]
    return SBiTransformer(
        input_size=cfg["input_size"], seq_len=cfg["seq_len"], num_hidden=cfg["num_hidden"],
        ffn_hidden=cfg["ffn_hidden"], n_heads=cfg["n_heads"],
        encoder_layers=cfg["encoder_layers"], dropout=cfg["dropout"],
        bilstm_size=cfg["bilstm_size"], num_layers=cfg["num_layers"],
        block_size=cfg["block_size"],
        use_transformer=ut, use_sparse=us, use_bilstm=ub)
