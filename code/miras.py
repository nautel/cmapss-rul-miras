"""Ho mo hinh Miras cho du doan RUL.

Paper: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni,
"It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias,
Retention, and Online Optimization", arXiv:2504.13173 (Google Research, 2025).

Miras coi moi sequence model la mot BO NHO LIEN KET hoc anh xa key->value tai thoi
diem suy dien, xac dinh boi 4 lua chon: (1) cau truc bo nho, (2) attentional bias
(ham muc tieu cua bo nho), (3) retention gate (co che quen), (4) thuat toan hoc.

Moi bien the trong file nay = mot to hop cua 4 lua chon do, dung CUNG mot vong lap,
nen so sanh giua chung co lap dung thu paper muon nghien cuu.

| bien the        | attentional bias        | retention gate                    | muc |
|-----------------|-------------------------|-----------------------------------|-----|
| linear_attn     | dot product (Hebbian)   | alpha = 1                         | Eq.8  |
| mamba2          | dot product             | alpha_t phu thuoc du lieu         | Eq.8  |
| deltanet        | l2 (delta rule)         | alpha = 1                         | Eq.9  |
| gated_deltanet  | l2                      | alpha_t phu thuoc du lieu         | Eq.9  |
| titans          | l2                      | alpha_t + momentum                | §4    |
| moneta          | l_p, p=3                | alpha_t + chuan hoa l_q, q=4      | Eq.24 |
| yaad            | Huber (tron l2 / l1)    | alpha_t                           | Eq.26 |
| memora          | l2                      | KL / softmax (elastic mem)        | Eq.27 |
| elastic         | l2                      | elastic net, soft-threshold       | Eq.22 |
| robust          | l2 + Delta*||e|| (worst case) | alpha_t                     | §5.1 v3 |

Huan luyen song song theo chunk dung nhu §5.4: gradient cua moi token trong chunk
tinh voi trang thai bo nho o DAU chunk. Nho vay ca chunk gom thanh matmul; chi con
T/b buoc tuan tu. `chunk = 1` la truy hoi CHINH XAC (khong xap xi).

CANH BAO ve xap xi chunk tren chuoi ngan: chunk dau tien bat dau voi W = 0 nen
e_i = W0 k_i - v_i = -v_i cho moi token trong chunk — tuc delta rule, l_p, Huber,
robust deu suy bien ve mot ham co dinh cua -v, va `deltanet` TRUNG voi `linear_attn`.
Paper dung chunk << T (T = 4096); o day T = 40-60 nen chunk 15-20 anh huong 1/3-1/2
chuoi. `run_miras_fix.sh` do anh huong nay bang cach quet chunk 1 / 5 / 20.

Retention gate bi chan duoi alpha >= 0.85: trong cong thuc chunk co ti so beta_t/beta_i
= prod alpha_j, neu alpha nho thi ti so nay bung len (1/alpha)^b. Voi b = 15 va
alpha >= 0.85 thi ti so <= 11, an toan trong float32. alpha = 0.85 van quen du nhanh
(0.85^45 = 6e-4 sau ca chuoi).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def _sign(x, sharp=10.0):
    """Sign(x) ~ tanh(alpha x) — xap xi tron, Remark 5 cua paper."""
    return torch.tanh(sharp * x)


def _abs(x):
    """|x| = sqrt(x^2 + eps) — xap xi tron, Remark 5."""
    return torch.sqrt(x * x + EPS)


class CausalDWConv(nn.Module):
    """Depthwise-separable conv1d nhan qua sau moi phep chieu q/k/v (§5.4, kernel 4)."""

    def __init__(self, d, k=4):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(d, d, k, groups=d, bias=False)

    def forward(self, x):                                  # (B,T,D)
        y = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return self.conv(y).transpose(1, 2)


class MirasLayer(nn.Module):
    """Mot lop Miras voi bo nho tuyen tinh W in R^{d x d}, M(W,k) = W k."""

    NEEDS_GATE = {"mamba2", "gated_deltanet", "titans", "moneta", "yaad",
                  "memora", "elastic", "robust"}

    def __init__(self, d_model, variant="gated_deltanet", chunk=15, dropout=0.0,
                 rank=0, causal=True):
        """`rank > 0`: bo nho hang thap W in R^{d x rank} thay vi d x d.

        Day la lua chon so 1 cua Miras (cau truc bo nho) day theo huong NGUOC voi
        paper: ho mo rong (MLP sau, he so 4), o day thu that. Ly do: TSHAE dung
        latent 2 chieu va STA-HPINN dung 3 — hai nguon doc lap cung ep bieu dien
        qua nut that rat hep, va do la thu duy nhat cac phuong phap manh co chung
        ma paper 1, ho Miras va cau hinh autoresearch deu khong co.
        """
        super().__init__()
        assert variant in ("linear_attn", "retnet", "mamba2", "deltanet",
                           "gated_deltanet", "titans", "moneta", "yaad",
                           "memora", "elastic", "robust")
        self.v, self.d, self.b = variant, d_model, chunk
        # causal=False: truc khong co thu tu (vd cam bien) — bo conv nhan qua, mask day du,
        # khong quen, khong momentum. Ban cu dung mask tam giac tren truc cam bien -> cam
        # bien i chi thay cam bien < i theo chi so tuy y. Sua 09-2026.
        self.causal = causal
        self.rank = rank if rank and rank < d_model else 0
        if self.rank:
            self.proj_k = nn.Linear(d_model, self.rank, bias=False)
            self.proj_q = nn.Linear(d_model, self.rank, bias=False)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.cq, self.ck, self.cv = (CausalDWConv(d_model) for _ in range(3))
        self.gate = nn.Linear(d_model, d_model)            # cong dau ra (§5.4)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # eta_t: buoc hoc cua bo nho, phu thuoc du lieu
        self.p_eta = nn.Linear(d_model, 1)
        # alpha_t: retention gate
        if variant == "retnet":
            self.log_alpha = nn.Parameter(torch.tensor(-0.05))
        elif variant in self.NEEDS_GATE:
            self.p_alpha = nn.Linear(d_model, 1)
        if variant == "titans":
            self.p_theta = nn.Linear(d_model, 1)           # he so momentum
        if variant == "yaad":
            self.p_delta = nn.Linear(d_model, 1)           # nguong Huber
        if variant == "robust":
            self.log_Delta = nn.Parameter(torch.tensor(-2.0))
        if variant == "elastic":
            self.log_gamma = nn.Parameter(torch.tensor(-3.0))   # nguong hard-forget
        if variant == "moneta":
            self.p, self.q = 3.0, 4.0
        if variant == "memora":
            self.c = nn.Parameter(torch.tensor(float(d_model)))  # ||W||_1 = c

    # ---- attentional bias: tra ve g_i sao cho grad = g_i k_i^T ----
    def _grad_dir(self, W0, K, V, delta=None):
        if self.v in ("linear_attn", "retnet", "mamba2"):
            return -V                                      # dot product (Hebbian)
        E = torch.einsum("bij,btj->bti", W0, K) - V         # e_i = W0 k_i - v_i
        if self.v == "moneta":
            return self.p * _sign(E) * _abs(E).pow(self.p - 1.0)
        if self.v == "yaad":                                # Eq. (16)/(26)
            n = E.norm(dim=-1, keepdim=True)
            return torch.where(n <= delta, E, delta * _sign(E))
        if self.v == "robust":                              # §5.1 variant 3
            D = self.log_Delta.exp()
            return E + D * E / (E.norm(dim=-1, keepdim=True) + EPS)
        return E                                            # l2 / delta rule

    def forward(self, x):
        B, T, D = x.shape
        b = self.b
        assert T % b == 0, f"T={T} phai chia het cho chunk={b}"
        if self.causal:
            Q = F.normalize(self.cq(self.wq(x)), dim=-1)    # l2-norm q,k (§5.4)
            K = F.normalize(self.ck(self.wk(x)), dim=-1)
            V = self.cv(self.wv(x))
        else:                               # truc khong co thu tu (cam bien): bo conv nhan qua
            Q = F.normalize(self.wq(x), dim=-1)
            K = F.normalize(self.wk(x), dim=-1)
            V = self.wv(x)
        if self.rank:                       # nut that: khoa/truy van song trong R^rank
            Q = F.normalize(self.proj_q(Q), dim=-1)
            K = F.normalize(self.proj_k(K), dim=-1)
        Dk = K.shape[-1]

        eta = torch.sigmoid(self.p_eta(x))                  # (B,T,1) in (0,1)
        if self.v in ("linear_attn", "deltanet") or not self.causal:
            # khong nhan qua: khong co "truoc/sau" nen khong co quen theo thoi gian
            alpha = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        elif self.v == "retnet":
            alpha = self.log_alpha.exp().clamp(0.85, 1.0).expand(B, T, 1)
        else:                                               # alpha_t in [0.85, 1]
            alpha = 0.85 + 0.15 * torch.sigmoid(self.p_alpha(x))
        theta = torch.sigmoid(self.p_theta(x)) if self.v == "titans" else None
        delta = F.softplus(self.p_delta(x)) + EPS if self.v == "yaad" else None

        idx = torch.arange(b, device=x.device)
        if self.causal:
            tri = (idx[:, None] >= idx[None, :]).to(x.dtype)  # mask nhan qua trong chunk
        else:                                                 # moi token thay moi token
            tri = torch.ones(b, b, device=x.device, dtype=x.dtype)
        momentum = self.v == "titans" and self.causal

        if self.v == "memora":
            W = torch.full((B, D, Dk), 1.0 / (D * Dk), device=x.device, dtype=x.dtype)
            W = W * self.c
        else:
            W = torch.zeros(B, D, Dk, device=x.device, dtype=x.dtype)
        S = torch.zeros_like(W) if momentum else None

        outs = []
        for c0 in range(0, T, b):
            sl = slice(c0, c0 + b)
            Kc, Vc, Qc = K[:, sl], V[:, sl], Q[:, sl]
            ec, ac = eta[:, sl], alpha[:, sl]

            # beta_t = prod_{j<=t} alpha_j  (trong chunk), va ti so beta_t / beta_j
            logb = torch.cumsum(torch.log(ac.squeeze(-1) + EPS), dim=1)   # (B,b)
            beta = logb.exp()                                             # (B,b)
            R = (logb[:, :, None] - logb[:, None, :]).exp() * tri         # (B,b,b)
            if momentum:                                                  # them momentum
                tc = theta[:, sl].squeeze(-1)
                lt = torch.cumsum(torch.log(tc + EPS), dim=1)
                Theta = lt.exp()                                          # prod_{j<=t} theta_j
                Mm = (lt[:, :, None] - lt[:, None, :]).exp() * tri
                C = R @ Mm
                # he so cua momentum mang tu chunk truoc: cS_t = sum_{j<=t} R[t,j] Theta_j
                cS = (R @ Theta.unsqueeze(-1)).squeeze(-1)                # (B,b)
            else:
                C = R

            G = self._grad_dir(W, Kc, Vc, delta[:, sl] if delta is not None else None)
            Gw = ec * G                                                   # eta_i * g_i

            # o_t = beta_t (W q_t) - sum_i C[t,i] (q_t . k_i) eta_i g_i
            base = torch.einsum("bij,btj->bti", W, Qc) * beta[..., None]
            if momentum:                                                  # + cS_t (S_0 q_t)
                base = base + torch.einsum("bij,btj->bti", S, Qc) * cS[..., None]
            A = (Qc @ Kc.transpose(1, 2)) * C
            outs.append(base - A @ Gw)

            # W_end = beta_b W - sum_i C[b,i] eta_i g_i k_i^T
            w = (C[:, -1, :, None] * Gw)                                  # (B,b,dv)
            Wn = W * beta[:, -1, None, None] - torch.einsum("btv,btk->bvk", w, Kc)
            if momentum:
                Wn = Wn + S * cS[:, -1, None, None]

            if self.v == "moneta":                                        # Eq. (24)
                nq = Wn.flatten(1).norm(p=self.q, dim=1).clamp_min(EPS)
                Wn = Wn / nq.pow(self.q - 2.0)[:, None, None]
            elif self.v == "memora":                                      # Eq. (27)
                # Wn - W*beta = -sum eta g k^T = -eta*grad  =>  z = a*log W - eta*grad.
                # (Ban cu tru them lan nua -> +eta*grad -> di LEN gradient. Sua 09-2026.)
                lw = torch.log(W.clamp_min(EPS))
                z = ac[:, -1, :, None] * lw + (Wn - W * beta[:, -1, None, None])
                Wn = self.c.abs() * torch.softmax(z.flatten(1), dim=1).view_as(Wn)
            elif self.v == "elastic":                                     # Eq. (22)
                g = self.log_gamma.exp()
                Wn = _sign(Wn) * F.relu(_abs(Wn) - g)
            if momentum:
                # momentum qua ranh gioi chunk: S_end = Theta_b S_0 - sum Mm[b,i] eta g k^T
                # (Ban cu tinh S roi bo di -> momentum reset moi chunk. Sua 09-2026.)
                wS = Mm[:, -1, :, None] * Gw
                S = S * Theta[:, -1, None, None] - torch.einsum("btv,btk->bvk", wS, Kc)
            W = Wn

        o = torch.cat(outs, dim=1)
        o = self.norm(o) * torch.sigmoid(self.gate(x))                    # gated output
        return self.drop(self.wo(o))


class MemoryMLP(nn.Module):
    """Bo nho SAU 2 tang M(k) = W1 gelu(W2 k) — dung cho titans_mlp.

    Gradient tinh bang tay (khong autograd long nhau) de chay duoc 45 buoc tuan tu.
    """

    def __init__(self, d, hidden):
        super().__init__()
        self.d, self.h = d, hidden
        self.W2_0 = nn.Parameter(torch.randn(hidden, d) / math.sqrt(d))
        self.W1_0 = nn.Parameter(torch.randn(d, hidden) / math.sqrt(hidden))

    def init(self, B):
        return (self.W1_0.expand(B, -1, -1).contiguous(),
                self.W2_0.expand(B, -1, -1).contiguous())

    @staticmethod
    def _gelu_d(z):
        c = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return c + z * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    def apply(self, W, k):
        W1, W2 = W
        z = torch.einsum("bhd,bd->bh", W2, k)
        return torch.einsum("bdh,bh->bd", W1, F.gelu(z)), z

    def grads(self, W, k, v):
        W1, W2 = W
        y, z = self.apply(W, k)
        a = F.gelu(z)
        e = y - v
        g1 = torch.einsum("bd,bh->bdh", e, a)
        dh = torch.einsum("bdh,bd->bh", W1, e) * self._gelu_d(z)
        g2 = torch.einsum("bh,bd->bhd", dh, k)
        return (g1, g2)


class TitansMLPLayer(nn.Module):
    """Titans-LMM dung nhu §4: bo nho MLP sau + l2 bias + momentum + weight decay.

    Chay TUAN TU tung buoc (khong chunk) vi bo nho khong tuyen tinh -> khong gom
    duoc thanh matmul. Dung de kiem chung xem bo nho SAU co giup gi cho chuoi 45 buoc.
    """

    def __init__(self, d_model, hidden_mult=2, dropout=0.0):
        super().__init__()
        self.d = d_model
        self.mem = MemoryMLP(d_model, d_model * hidden_mult)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.cq, self.ck, self.cv = (CausalDWConv(d_model) for _ in range(3))
        self.p_eta = nn.Linear(d_model, 1)
        self.p_alpha = nn.Linear(d_model, 1)
        self.p_theta = nn.Linear(d_model, 1)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        Q = F.normalize(self.cq(self.wq(x)), dim=-1)
        K = F.normalize(self.ck(self.wk(x)), dim=-1)
        V = self.cv(self.wv(x))
        eta = torch.sigmoid(self.p_eta(x))
        alpha = 0.5 + 0.5 * torch.sigmoid(self.p_alpha(x))
        theta = torch.sigmoid(self.p_theta(x))

        W = self.mem.init(B)
        S = tuple(torch.zeros_like(w) for w in W)
        outs = []
        for t in range(T):
            g = self.mem.grads(W, K[:, t], V[:, t])
            S = tuple(theta[:, t, :, None] * s - eta[:, t, :, None] * gi
                      for s, gi in zip(S, g))
            W = tuple(alpha[:, t, :, None] * w + s for w, s in zip(W, S))
            outs.append(self.mem.apply(W, Q[:, t])[0])
        o = torch.stack(outs, 1)
        o = self.norm(o) * torch.sigmoid(self.gate(x))
        return self.drop(self.wo(o))


class MirasBlock(nn.Module):
    """Lop Miras + FFN, moi phan co residual + LayerNorm (giong khoi encoder cua paper 1)."""

    def __init__(self, d_model, ffn_hidden, variant, dropout, chunk, rank=0, causal=True):
        super().__init__()
        if variant == "titans_mlp":
            self.mix = TitansMLPLayer(d_model, dropout=dropout)
        else:
            self.mix = MirasLayer(d_model, variant, chunk, dropout, rank, causal)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_hidden), nn.ReLU(),
                                 nn.Linear(ffn_hidden, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.mix(x))
        return self.n2(x + self.drop(self.ffn(x)))


class MirasRUL(nn.Module):
    """Backbone du doan RUL: giu y nguyen phan ngoai cua paper 1 (chieu tuyen tinh +
    position embedding hoc duoc -> N lop -> FC tren buoc cuoi) va CHI thay khoi
    attention bang lop Miras, de so sanh cong bang."""

    def __init__(self, input_size=17, seq_len=45, num_hidden=16, ffn_hidden=32,
                 layers=3, dropout=0.2, variant="gated_deltanet", chunk=15,
                 bilstm_size=0, bilstm_layers=2, rank=0, sensor_branch=False,
                 bottleneck=0, reg_dims=64):
        """`sensor_branch`: chay them mot nhanh Miras tren chieu CAM BIEN (moi cam
        bien la mot token) — y muon tu STA-HPINN, noi hai nhanh attention song song
        la thanh phan chinh. Miras nguyen ban chi tac dong len chieu thoi gian.

        `bottleneck > 0`: ep trang thai cuoi qua nut that hep truoc khi hoi quy,
        de lay `z` cho triplet loss (xem train_miras2.py).
        """
        super().__init__()
        self.proj = nn.Linear(input_size, num_hidden)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, num_hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MirasBlock(num_hidden, ffn_hidden, variant, dropout, chunk, rank)
            for _ in range(layers)])
        self.sensor_branch = sensor_branch
        if sensor_branch:
            # moi cam bien la mot token; chunk = so cam bien -> mot chunk duy nhat
            self.s_proj = nn.Linear(seq_len, num_hidden)
            self.s_pos = nn.Parameter(torch.zeros(1, input_size, num_hidden))
            nn.init.trunc_normal_(self.s_pos, std=0.02)
            self.s_blocks = nn.ModuleList([
                MirasBlock(num_hidden, ffn_hidden, variant, dropout, input_size, rank,
                           causal=False)
                for _ in range(layers)])
        # bilstm_size > 0: gan them dau BiLSTM giong paper 1, de so sanh o cung
        # dung luong tham so (backbone Miras thuan chi ~9k tham so, SBi ~49k).
        self.bilstm = (nn.LSTM(num_hidden, bilstm_size, num_layers=bilstm_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if bilstm_layers > 1 else 0.0)
                       if bilstm_size else None)
        d_out = 2 * bilstm_size if bilstm_size else num_hidden
        if sensor_branch:
            d_out += num_hidden
        self.bottleneck = bottleneck
        if bottleneck:
            self.to_z = nn.Linear(d_out, bottleneck)
            self.head = nn.Sequential(nn.Tanh(), nn.Dropout(dropout),
                                      nn.Linear(bottleneck, reg_dims), nn.Tanh(),
                                      nn.Linear(reg_dims, 1))
        else:
            self.head = nn.Linear(d_out, 1)

    def forward(self, x, return_z=False):
        h = self.drop(self.proj(x) + self.pos)
        for blk in self.blocks:
            h = blk(h)
        if self.bilstm is not None:
            h, _ = self.bilstm(h)
        f = h[:, -1]
        if self.sensor_branch:
            g = self.drop(self.s_proj(x.transpose(1, 2)) + self.s_pos)
            for blk in self.s_blocks:
                g = blk(g)
            f = torch.cat([f, g.mean(1)], dim=1)
        z = self.to_z(f) if self.bottleneck else f
        y = self.head(z).squeeze(-1)
        return (y, z) if return_z else y


_BASE = ["linear_attn", "retnet", "mamba2", "deltanet", "gated_deltanet",
         "titans", "moneta", "yaad", "memora", "elastic", "robust", "titans_mlp"]
VARIANTS = _BASE + [v + "+bilstm" for v in _BASE]


def build_miras(cfg, variant):
    """`variant` co hau to "+bilstm" thi gan them dau BiLSTM cua paper 1."""
    bl = variant.endswith("+bilstm")
    return MirasRUL(input_size=cfg["input_size"], seq_len=cfg["seq_len"],
                    num_hidden=cfg["num_hidden"], ffn_hidden=cfg["ffn_hidden"],
                    layers=cfg["encoder_layers"], dropout=cfg["dropout"],
                    variant=variant.replace("+bilstm", ""), chunk=cfg.get("chunk", 15),
                    bilstm_size=cfg["bilstm_size"] if bl else 0,
                    bilstm_layers=cfg["num_layers"],
                    rank=cfg.get("mem_rank", 0),
                    sensor_branch=bool(cfg.get("sensor_branch", 0)),
                    bottleneck=cfg.get("bottleneck", 0),
                    reg_dims=cfg.get("reg_dims", 64))
