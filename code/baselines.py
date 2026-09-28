"""Cac baseline kinh dien cua van lieu C-MAPSS, cai trong CUNG pipeline.

Ly do phai tu chay thay vi trich so tu paper: so cong bo cua paper 1 khong tai lap
duoc (11,43 so voi 13,96 do duoc), nen lay no lam moc la so hai thu khac dieu kien.
Muon khang dinh hon kem thi moi phuong phap phai chay tren cung tien xu ly, cung
tap val, cung nhan, cung so seed.

| ten      | nguon |
|----------|-------|
| `dcnn`   | Li, Ding, Sun, "RUL estimation in prognostics using deep convolution neural networks", RESS 172 (2018) |
| `lstm`   | Zheng et al., "Long short-term memory network for remaining useful life estimation", ICPHM 2017 |
| `bilstm` | Wang et al., bidirectional LSTM cho RUL |
| `gru`    | bien the GRU cua tren |
| `tcn`    | Bai et al., temporal convolutional network (dilated causal conv) |
| `cnn_lstm`| ho lai CNN + LSTM (Remadna 2020, Che 2021) |
| `mlp`    | can duoi: bo qua cau truc thoi gian, duoi phang cua so |
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNN(nn.Module):
    """Li et al. 2018: 4 lop conv 10 filter kernel (10x1) + 1 lop conv 1 filter (3x1)."""

    def __init__(self, input_size, seq_len, n_filters=10, k=10, dropout=0.5, fc=100):
        super().__init__()
        pad = k // 2
        self.convs = nn.ModuleList()
        c_in = 1
        for _ in range(4):
            self.convs.append(nn.Conv2d(c_in, n_filters, (k, 1), padding=(pad, 0)))
            c_in = n_filters
        self.last = nn.Conv2d(n_filters, 1, (3, 1), padding=(1, 0))
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(seq_len * input_size, fc)
        self.fc2 = nn.Linear(fc, 1)

    def forward(self, x):                       # (B,T,D)
        h = x.unsqueeze(1)                      # (B,1,T,D)
        for c in self.convs:
            h = torch.tanh(c(h))[:, :, :x.shape[1]]
        h = torch.tanh(self.last(h))[:, :, :x.shape[1]]
        h = self.drop(h.flatten(1))
        return self.fc2(torch.tanh(self.fc1(h))).squeeze(-1)


class RNNBaseline(nn.Module):
    """LSTM / BiLSTM / GRU + hai lop FC, lay trang thai an cuoi cung."""

    def __init__(self, input_size, hidden=64, layers=2, kind="lstm",
                 bidir=False, dropout=0.2, fc=8):
        super().__init__()
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cls(input_size, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidir, dropout=dropout if layers > 1 else 0.0)
        d = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.Linear(d, fc), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(fc, 1))

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1]).squeeze(-1)


class TCN(nn.Module):
    """Dilated causal conv, kernel 3, dilation 1/2/4/8 + residual."""

    def __init__(self, input_size, ch=32, k=3, dilations=(1, 2, 4, 8), dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.res = nn.ModuleList()
        c_in = input_size
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(c_in, ch, k, dilation=d), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(ch, ch, k, dilation=d), nn.ReLU(), nn.Dropout(dropout)))
            self.res.append(nn.Conv1d(c_in, ch, 1) if c_in != ch else nn.Identity())
            c_in = ch
        self.k, self.dil = k, dilations
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        h = x.transpose(1, 2)                   # (B,D,T)
        for blk, res, d in zip(self.blocks, self.res, self.dil):
            pad = (self.k - 1) * d
            y = h
            for layer in blk:
                if isinstance(layer, nn.Conv1d):
                    y = layer(F.pad(y, (pad, 0)))
                else:
                    y = layer(y)
            h = y + res(h)
        return self.head(h[:, :, -1]).squeeze(-1)


class CNNLSTM(nn.Module):
    """Conv1d rut dac trung cuc bo -> LSTM mo hinh hoa thoi gian."""

    def __init__(self, input_size, ch=32, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU())
        self.rnn = nn.LSTM(ch, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.head(h[:, -1]).squeeze(-1)


class MLPFlat(nn.Module):
    """Can duoi: duoi phang cua so, bo qua hoan toan cau truc thoi gian."""

    def __init__(self, input_size, seq_len, hidden=100, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * input_size, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


BASELINES = ["dcnn", "lstm", "bilstm", "gru", "tcn", "cnn_lstm", "mlp"]


def build_baseline(cfg, name):
    d, L, do = cfg["input_size"], cfg["seq_len"], cfg["dropout"]
    h = cfg.get("bl_hidden", 64)
    if name == "dcnn":
        return DCNN(d, L, dropout=max(do, 0.5))
    if name in ("lstm", "bilstm", "gru"):
        return RNNBaseline(d, hidden=h, layers=cfg.get("num_layers", 2),
                           kind="gru" if name == "gru" else "lstm",
                           bidir=(name == "bilstm"), dropout=do)
    if name == "tcn":
        return TCN(d, ch=cfg.get("ffn_hidden", 32), dropout=do)
    if name == "cnn_lstm":
        return CNNLSTM(d, ch=cfg.get("ffn_hidden", 32), hidden=h, dropout=do)
    if name == "mlp":
        return MLPFlat(d, L, hidden=100, dropout=do)
    raise ValueError(name)
