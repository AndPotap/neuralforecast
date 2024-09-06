from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter as Par


def gen_HiPPO_s4d(N: int):
    AB = torch.sqrt(1 + 2 * torch.arange(N))
    A = AB[:, None] * AB[None, :]
    A = torch.diag(torch.arange(N)) - torch.tril(A)
    P = torch.sqrt(torch.arange(N) + 0.5)

    S = A + P[:, None] * P[None, :]
    S_diag = torch.diagonal(S)
    Lam_re = torch.mean(S_diag) * torch.ones_like(S_diag)
    Lam_im, _ = torch.linalg.eigh(-S * 1j)

    return Lam_re, Lam_im


def init_hippo(D: int, N: int):
    A_re, A_im = gen_HiPPO_s4d(N)
    A_re = A_re[None, :, None].broadcast_to((D, N, 1)).clone()
    A_im = A_im[None, :, None].broadcast_to((D, N, 1)).clone()
    sigma = 0.5**0.5
    C = sigma * torch.randn(size=(D, N, 2))
    D = torch.ones((1, D, 1))
    return A_re, A_im, C, D


def compute_kernel_s4d_zoh(C: torch.Tensor, A: torch.Tensor, seq_len: int, step: torch.Tensor):
    H, N = A.shape[:2]
    times = torch.arange(seq_len, device=A.device)[None, None, :]
    part = torch.exp(times * (step * A))
    cons = (torch.exp(step * A) - 1) / A
    ker = C * cons * part
    return torch.sum(ker, dim=-2).real


def causal_conv(x: torch.Tensor, K: torch.Tensor):
    # x: [B,D,L]
    # K: [D,L] or [B,D,L]
    if len(K.shape) < 3:
        K = torch.reshape(K, (1,) + K.shape)  # [1,D,L]
    xd = F.pad(x, (0, K.shape[-1]), "constant")
    Kd = F.pad(K, (0, x.shape[-1]), "constant")
    xd = torch.fft.rfft(xd, dim=-1)
    Kd = torch.fft.rfft(Kd, dim=-1)
    out = xd * Kd
    out = torch.fft.irfft(out, dim=-1)[:, :, : x.shape[-1]]
    return out


class S4DLayer(nn.Module):
    def __init__(self, d_state: int, width: int, input_size: int, device=None):
        vars2parms = ["A_re", "A_im", "C", "D", "log_delta"]
        super().__init__()
        self.N, self.H, self.L = d_state, width, input_size
        self.A_re, self.A_im, self.C, self.D = init_hippo(self.H, self.N)
        log_delta_min, log_delta_max = torch.log(torch.tensor(0.001)), torch.log(torch.tensor(0.1))
        unif = torch.rand(size=(self.H, 1, 1))
        self.log_delta = unif * (log_delta_max - log_delta_min) + log_delta_min
        self.convert2parms(vars2parms, device)

    def convert2parms(self, vars2parms, device):
        all_vars = ["A_re", "A_im", "C", "D", "log_delta"]
        for var_name in all_vars:
            val = getattr(self, var_name)
            if var_name in vars2parms:
                setattr(self, var_name, Par(val))
            else:
                setattr(self, var_name, val.to(device))

    def forward(self, x):
        # x: [B,D,L]
        A = torch.clip(self.A_re, None, -1e-4) + 1j * self.A_im
        C = self.C[..., [0]] + 1j * self.C[..., [1]]
        K = compute_kernel_s4d_zoh(C, A, self.L, torch.exp(self.log_delta))
        x = causal_conv(x, K) + self.D * x
        return x  # [B,D,L]


class SeqBlockGLU(nn.Module):
    def __init__(self, seq_cls: nn.Module, seq_kwargs: Dict, width: int, dropout: float):
        super().__init__()
        self.seq = seq_cls(**seq_kwargs)
        self.ln = nn.LayerNorm(width)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp1 = nn.Linear(width, width)
        self.mlp2 = nn.Linear(width, width)
        init.zeros_(self.mlp1.bias)
        init.zeros_(self.mlp2.bias)

    def forward(self, x):
        # x: [B,L,D]
        skip = x
        x = self.seq(self.ln(x).transpose(-1, -2))
        x = self.dropout(F.gelu(x.transpose(-1, -2)))
        x = self.mlp1(x) * F.sigmoid(self.mlp2(x))
        out = skip + self.dropout(x)
        return out  # [B,L,D]


class SSM_Rev(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        num_horizons: int,
        d_state: int,
        width: int,
        depth: int,
        hist_size: int,
        futr_size: int,
        dropout: float,
        device,
    ):
        super().__init__()
        seq_kwargs = {"d_state": d_state, "input_size": input_size + num_horizons, "width": width, "device": device}
        self.input_layer = nn.Linear(1 + hist_size + futr_size, width)
        self.proj = nn.Linear(input_size, num_horizons)
        self.output_layer = nn.Linear(width, 1)
        self.mixer = nn.Linear(input_size + num_horizons, num_horizons)
        self.layers = nn.ModuleList()
        for ldx in range(depth):
            self.layers.append(SeqBlockGLU(seq_cls=S4DLayer, seq_kwargs=seq_kwargs, width=width, dropout=dropout))

    def forward(self, y, yhat, xt, xf) -> torch.Tensor:
        # y: [B,L]
        # yhat: [B,H]
        # xt: [B,L,Dt]
        # xf: [B,L+H,Df]
        y_all = torch.concat((y[:, :, None], yhat[:, :, None]), dim=-2)  # [B,L+H,1]
        xt_f = self.proj(xt.transpose(-1, -2)).transpose(-1, -2)  # [B,H,Dt]
        zt = torch.concat((xt, xt_f), dim=-2)  # [B,L+H,Dt]
        z_all = torch.concat((y_all, zt, xf), dim=-1)  # [B,L+H,1+Dt+Df]
        z = self.input_layer(z_all)  # [B,L+H,W]
        for layer in self.layers:
            z = layer(z)  # [B,L+H,W]
        z = self.output_layer(z)  # [B,L+H,1]
        yhat = self.mixer(z[:, :, 0])  # [B,H]
        return yhat  # [B,H]
