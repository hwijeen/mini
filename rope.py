import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.cache_initialized = False

    def _build_cache(self, x: torch.Tensor):
        B, nh, T, d = x.shape
        assert d % 2 == 0, "The input dimension must be even."
        # [1, 1, 2, ..., T]
        pos_range = torch.arange(1, T+1, device=x.device, dtype=x.dtype) # (T, )
        # [0, 1, 2, ... d/2]
        dim_range = torch.arange(d // 2, device=x.device, dtype=x.dtype) # (d/2, )
        theta = self.base ** (-2 * (dim_range -1) / d) # (d/2, )
        pos_theta = torch.einsum("T, d -> Td", pos_range, theta) # (T, d/2)
        self.cos = torch.cos(pos_theta) # (T, d/2)
        self.cos = torch.cat([self.cos, self.cos], dim=-1) # (T, d)
        self.sin = torch.sin(pos_theta) # (T, d/2)
        self.sin = torch.cat([self.sin, self.sin], dim=-1) # (T, d)
        self.cache_initialized = True

    def forward(self, x: torch.Tensor):
        if not self.cache_initialized:
            self._build_cache(x)
        x_ = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)  # interleave
        res = x * self.cos + x_ * self.sin
        return res
