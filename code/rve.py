"""RVE / TSHAE — variational encoder with an EXTREMELY NARROW bottleneck for RUL prediction.

Reference code: `external_code/Time-Series-Hybrid-Autoencoder` (insdout), an implementation
of the "variational encoding" approach (Costa & Sanchez, RESS 2022). Their original config:
LSTM hidden 300, **latent_dim = 2**, loss = Recon(1) + Reg(1) + **Triplet(150)** + KL(0).

Key idea, and what sets it apart from everything else run in this project: force the
representation through a 2-3-dim bottleneck, then constrain that latent space to have a
MONOTONIC STRUCTURE with respect to degradation level. STA-HPINN (arXiv:2405.12377) also
independently compresses down to 3 dims. Two independent sources, same conclusion —
whereas paper 1, the Miras family and the autoresearch config all go the opposite way
(adding capacity).

The triplet loss here uses **batch-hard mining** instead of a dataloader that pre-generates
pairs: within each batch, the positive is the sample with the closest RUL and the negative
is the sample with the farthest RUL. Same objective, without changing the data pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RVE(nn.Module):
    def __init__(self, input_size, seq_len, hidden=64, latent=2, layers=1,
                 bidirectional=True, dropout=0.2, reg_dims=100, reconstruct=True):
        super().__init__()
        self.latent, self.seq_len, self.input_size = latent, seq_len, input_size
        self.reconstruct = reconstruct
        nd = 2 if bidirectional else 1
        self.enc = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0.0)
        self.fc_mean = nn.Sequential(nn.Dropout(dropout), nn.Linear(nd * hidden, latent))
        self.fc_logvar = nn.Sequential(nn.Dropout(dropout), nn.Linear(nd * hidden, latent))
        self.reg = nn.Sequential(nn.Linear(latent, reg_dims), nn.Tanh(),
                                 nn.Dropout(dropout), nn.Linear(reg_dims, 1))
        if reconstruct:
            self.dec_lstm = nn.LSTM(latent, hidden, num_layers=layers, batch_first=True,
                                    bidirectional=bidirectional)
            self.dec_out = nn.Linear(nd * hidden, input_size)

    def encode(self, x):
        _, (h, _) = self.enc(x)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.enc.bidirectional else h[-1]
        mean, logvar = self.fc_mean(h), self.fc_logvar(h)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if self.training else mean
        return z, mean, logvar

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        y = self.reg(z).squeeze(-1)
        xh = None
        if self.reconstruct:
            h, _ = self.dec_lstm(z.unsqueeze(1).repeat(1, self.seq_len, 1))
            xh = self.dec_out(h)
        return y, xh, z, mean, logvar


def batch_hard_triplet(z, y, margin=0.4):
    """Within each batch: positive = closest RUL, negative = farthest RUL.

    Replaces the original's pair-generating dataloader; same objective — force distances
    in latent space to track distances in degradation level.
    """
    if len(z) < 3:
        return z.new_zeros(())
    dy = (y[:, None] - y[None, :]).abs()
    eye = torch.eye(len(y), device=y.device, dtype=torch.bool)
    dy_pos = dy.masked_fill(eye, float("inf"))
    pos = dy_pos.argmin(1)
    neg = dy.masked_fill(eye, -1.0).argmax(1)
    return F.triplet_margin_loss(z, z[pos], z[neg], margin=margin, p=2)


def build_rve(cfg):
    return RVE(input_size=cfg["input_size"], seq_len=cfg["seq_len"],
               hidden=cfg.get("rve_hidden", 64), latent=cfg.get("rve_latent", 2),
               layers=cfg.get("rve_layers", 1), dropout=cfg.get("dropout", 0.2),
               reg_dims=cfg.get("rve_reg", 100),
               reconstruct=bool(cfg.get("rve_recon", 1)))
