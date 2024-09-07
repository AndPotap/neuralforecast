import torch
import torch.nn as nn


class Flat_Rev(nn.Module):
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
        stat_size: int,
        dropout: float,
        device,
    ):
        super().__init__()
        in_size = (1 + futr_size) * (input_size + num_horizons)
        in_size += (hist_size + stat_size) * (input_size)
        self.input_layer = nn.Linear(in_size, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.output_layer = nn.Linear(width, num_horizons)

    def forward(self, y, yhat, xt, xf, xs) -> torch.Tensor:
        # y: [B,L]
        # yhat: [B,H]
        # xt: [B,L,D_t]
        # xf: [B,L+H,D_f]
        # xs: [B,D_s]
        B, H = yhat.shape
        y_all = torch.concat((y[:, :, None], yhat[:, :, None]), dim=-2)
        z_all = torch.concat((y_all, xf), dim=-1)
        z_all = z_all.reshape(B, -1)
        if xt is not None:
            z_all = torch.concat((z_all, xt.reshape(B, -1)), dim=-1)
        if xs is not None:
            z_all = torch.concat((z_all, xs.reshape(B, -1)), dim=-1)
        z = self.input_layer(z_all)
        for layer in self.layers:
            z = layer(z)
        z = self.output_layer(z)
        yhat = z.reshape(B, H)
        return yhat  # [B,H]
