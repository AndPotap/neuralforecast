from typing import Dict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter as Par

from ..losses.pytorch import MAE
from ..common._base_windows import BaseWindows
from ..common._mlp import Flat_Rev


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


class S4_backbone(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        num_horizons: int,
        d_state: int,
        width: int,
        depth: int,
        dropout: float,
        device,
    ):
        super().__init__()
        seq_kwargs = {"d_state": d_state, "input_size": input_size, "width": width, "device": device}
        self.input_layer = nn.Linear(1, width)
        self.output_layer = nn.Linear(width, 1)
        self.mixer = nn.Linear(input_size, num_horizons)
        self.layers = nn.ModuleList()
        for ldx in range(depth):
            self.layers.append(SeqBlockGLU(seq_cls=S4DLayer, seq_kwargs=seq_kwargs, width=width, dropout=dropout))

    def forward(self, xt_target: torch.Tensor, xt: torch.Tensor, xs: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        # xt_target: [B*W,L,1]
        yt = self.input_layer(xt_target)  # [B*W,L,D]
        for layer in self.layers:
            yt = layer(yt)  # [B*W,L,D]
        yt = self.output_layer(yt)  # [B*W,L,1]
        yt = self.mixer(yt[:, :, 0])  # [B*W,H]
        return yt  # [B*W,H]


class S4_Flat(BaseWindows):
    def __init__(
        self,
        h,
        input_size,
        d_state,
        width,
        depth,
        dropout,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = -1,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader=0,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        **trainer_kwargs,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            windows_batch_size=windows_batch_size,
            valid_batch_size=valid_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            **trainer_kwargs,
        )
        self.flat = Flat_Rev(
            input_size=input_size,
            num_horizons=h,
            d_state=d_state,
            width=width,
            depth=depth,
            dropout=dropout,
            hist_size=self.hist_exog_size,
            futr_size=self.futr_exog_size,
            stat_size=self.stat_exog_size,
            device=None,
        )
        self.model = S4_backbone(
            input_size=input_size,
            num_horizons=h,
            d_state=d_state,
            width=width,
            depth=depth,
            dropout=dropout,
            device=None,
        )

    def forward(self, windows_batch):
        xt_target = windows_batch["insample_y"]
        xt = windows_batch["hist_exog"]
        xs = windows_batch["stat_exog"]
        xf = windows_batch["futr_exog"]
        forecast = self.model(xt_target=xt_target[:, :, None], xt=xt, xs=xs, xf=xf)
        forecast = self.flat(y=xt_target, yhat=forecast, xt=xt, xf=xf, xs=xs)
        return forecast
