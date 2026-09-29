"""STA-HPINN re-implementation — Spatio-temporal Attention-based Hidden Physics-informed NN.

Source: Feilong Jiang et al., arXiv:2405.12377 (2024). This is the most recent SOTA result
found that reports both RMSE + Score on all 4 C-MAPSS subsets: 11.27 / 13.21 / 8.30 / 13.31.

Architecture (Section 2 of the paper):
  Encoder = embedding -> TWO parallel attention branches -> feature fusion
     sensor branch: X^T (S,T) -> Linear(T->D) -> self-attn over the S axis -> (S,D)
     time branch  : X   (T,S) -> Linear(S->D) -> self-attn over the T axis -> (T,D)
     fusion: concat to (T+S, D) -> Conv kernel (T+S)x1, 3 channels -> SENet -> Linear
             -> H (3-dim)
  AHPINN:
     MLP  : (H, t) -> RUL            (3 hidden layers x 10 neurons, BatchNorm, tanh)
     NFNN : (H, dRUL/dH, d2RUL/dH2, d3RUL/dH3) -> N     (with self-attention)
     physics residual f = dRUL/dt - N ; loss = l1*L_data + l2*L_f
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnBlock(nn.Module):
    """Self-attention + FFN, each with residual + LayerNorm (Fig. 2)."""

    def __init__(self, d, n_heads=1, ffn=None, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        ffn = ffn or 4 * d
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.ReLU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.att(x, x, x, need_weights=False)
        x = self.n1(x + self.drop(a))
        return self.n2(x + self.drop(self.ffn(x)))


class SENet(nn.Module):
    """Squeeze-and-excitation over channels (the 3 channels of the fusion block)."""

    def __init__(self, ch, r=1):
        super().__init__()
        h = max(1, ch // r)
        self.fc = nn.Sequential(nn.Linear(ch, h), nn.ReLU(), nn.Linear(h, ch), nn.Sigmoid())

    def forward(self, x):                      # (B,C,L)
        w = self.fc(x.mean(-1))
        return x * w.unsqueeze(-1)


class Encoder(nn.Module):
    def __init__(self, seq_len, n_sensor, d=32, n_heads=1, hidden=3, dropout=0.1):
        super().__init__()
        self.emb_s = nn.Linear(seq_len, d)     # sensor branch: one token per sensor
        self.emb_t = nn.Linear(n_sensor, d)    # time branch: one token per time step
        self.att_s = AttnBlock(d, n_heads, dropout=dropout)
        self.att_t = AttnBlock(d, n_heads, dropout=dropout)
        self.conv = nn.Conv1d(1, 3, kernel_size=seq_len + n_sensor, stride=1)
        self.se = SENet(3)
        self.out = nn.Linear(3 * d, hidden)

    def forward(self, x):                      # (B,T,S)
        fs = self.att_s(self.emb_s(x.transpose(1, 2)))     # (B,S,D)
        ft = self.att_t(self.emb_t(x))                     # (B,T,D)
        fr = torch.cat([ft, fs], dim=1)                    # (B,T+S,D)
        B, L, D = fr.shape
        # conv kernel (T+S)x1: collapse the (T+S) axis to 1, keep D, output 3 channels
        h = self.conv(fr.transpose(1, 2).reshape(B * D, 1, L))     # (B*D,3,1)
        h = self.se(h).reshape(B, D, 3).transpose(1, 2).reshape(B, 3 * D)
        return self.out(h)                                 # (B,hidden)


def mlp(sizes, act=nn.Tanh, bn=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if bn:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


class NFNN(nn.Module):
    """Network learning the nonlinear function N, enhanced with self-attention (Fig. 4)."""

    def __init__(self, n_in, d=16, n_hidden=10, layers=3):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.att = AttnBlock(d, 1, ffn=2 * d, dropout=0.0)
        self.head = mlp([n_in * d] + [n_hidden] * layers + [1], bn=False)

    def forward(self, z):                      # (B,n_in)
        h = self.att(self.proj(z.unsqueeze(-1)))
        return self.head(h.flatten(1))


class STAHPINN(nn.Module):
    def __init__(self, seq_len, n_sensor, d=32, hidden=3, n_hidden=10, layers=3,
                 n_heads=1, dropout=0.1, cap=125.0):
        super().__init__()
        self.enc = Encoder(seq_len, n_sensor, d, n_heads, hidden, dropout)
        self.mlp = mlp([hidden + 1] + [n_hidden] * layers + [1], bn=True)
        self.nfnn = NFNN(4 * hidden)
        self.hidden, self.cap = hidden, cap

    def rul_from(self, H, t):
        return self.mlp(torch.cat([H, t], dim=1))

    def forward(self, x, t=None, physics=False):
        if t is None:                          # pure inference: no derivatives needed
            H = self.enc(x)
            z = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            return self.rul_from(H, z).squeeze(-1)
        H = self.enc(x)
        t = t.reshape(-1, 1)
        if not physics:
            return self.rul_from(H, t).squeeze(-1)
        H = H.requires_grad_(True)
        t = t.requires_grad_(True)
        u = self.rul_from(H, t)
        g = lambda y, v: torch.autograd.grad(y.sum(), v, create_graph=True)[0]
        d1 = g(u, H)                            # dRUL/dH        (B,hidden)
        d2 = g(d1, H)                           # d2RUL/dH2
        d3 = g(d2, H)                           # d3RUL/dH3
        dt = g(u, t)                            # dRUL/dt        (B,1)
        N = self.nfnn(torch.cat([H, d1, d2, d3], dim=1))
        return u.squeeze(-1), (dt - N).squeeze(-1)      # (RUL, physics residual f)


class ReLoBRaLo:
    """Relative Loss Balancing with Random Lookback (Bischof & Kraus 2021) — adjusts
    the weights between the data loss and the physics loss during training."""

    def __init__(self, n=2, alpha=0.999, tau=1.0, rho=0.999):
        self.n, self.alpha, self.tau, self.rho = n, alpha, tau, rho
        self.lam = None
        self.l0 = None
        self.lprev = None

    def __call__(self, losses):
        L = torch.stack([l.detach() for l in losses])
        if self.l0 is None:
            self.l0, self.lprev = L.clone(), L.clone()
            self.lam = torch.ones_like(L)
            return self.lam
        def bal(a, b):
            return torch.softmax((a / (b * self.tau + 1e-12)), 0) * self.n
        rho = 1.0 if torch.rand(1).item() < self.rho else 0.0
        lam_hat = rho * bal(L, self.lprev) + (1 - rho) * bal(L, self.l0)
        self.lam = self.alpha * self.lam + (1 - self.alpha) * lam_hat
        self.lprev = L.clone()
        return self.lam


def build_sta(cfg):
    return STAHPINN(seq_len=cfg["seq_len"], n_sensor=cfg["input_size"],
                    d=cfg.get("sta_d", 32), hidden=cfg.get("sta_hidden", 3),
                    n_hidden=cfg.get("sta_neurons", 10), layers=cfg.get("sta_layers", 3),
                    n_heads=cfg.get("n_heads", 1), dropout=cfg.get("dropout", 0.1))
