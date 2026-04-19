# reproduce from https://github.com/Anonymous1252022/EXAQ

import torch
import math
from typing import Optional

from torch.nn.functional import softmax
import numpy as np

# Table 1: linear approximation C* = a * σ + b
_OPT = {2: (-1.66, -1.85), 3: (-1.75, -2.06), 4: (-1.83, -2.27)}

class exaq:
    def __init__(self):
        self._clip_buf = []    # 校准缓冲
        self._min_buf  = []
        self._cal_steps = 25    # 校准步数

    @torch.no_grad()
    def exaq_softmax(self,
            input_: torch.Tensor,
            dim: int,
            mask: torch.Tensor,
            bitwidth: int = 3
    ) -> torch.Tensor:
        # --- Calibration buffers (instance attributes), fixed steps=25; no extra
        # --- grouping for batch=1 and bitwidth=2 scenarios ---
        # --- 1. Stabilization (mask invalid positions first to avoid
        # --- participation in amax/σ/quantization) ---
        input_ = input_.masked_fill(~mask, float("-inf"))
        x = input_ - input_.amax(dim=dim, keepdim=True)

        # --- 2. Compute σ, instantaneous C*, and instantaneous min (only over
        # --- valid positions) ---
        x_valid = x[mask]
        sigma   = x_valid.float().std(unbiased=False).item()
        a, b    = _OPT[bitwidth]  # fixed for 3-bit
        C_now   = a * sigma + b           # expected to be negative
        min_now = x_valid.min().item()    # minimum value for this sample (negative)

        # --- 2.1 Calibration: collect first 25 steps, then use the mean ---
        if len(self._clip_buf) >= self._cal_steps:
            C_use = float(np.mean(self._clip_buf))
        else:
            C_use = C_now
            self._clip_buf.append(float(C_now))

        if len(self._min_buf) >= self._cal_steps:
            min_use = float(np.mean(self._min_buf))
        else:
            min_use = min_now
            self._min_buf.append(float(min_now))

        # Numeric safeguards
        if not np.isfinite(C_use) or C_use >= 0.0:
            C_use = -1e-6

        # Lower bound is the larger of the two (closer to 0): [lo, 0]
        lo = max(C_use, min_use)

        # --- 3. One-sided quantization (endpoints representable):
        # --- Δ = (0 - lo) / (2^b - 1) ---
        L = 2 ** bitwidth  # fixed 3-bit
        delta = (0.0 - lo) / float(L - 1)

        x = x.clone()
        if not (np.isfinite(delta) and delta > 0.0):
            xv = x[mask].clamp_(min=lo, max=0.0)
        else:
            xv = x[mask].clamp_(min=lo, max=0.0)
            q  = torch.round((xv - lo) / delta).clamp_(0, L - 1)
            xv = lo + q * delta
        x[mask] = xv  # write back only valid positions; invalid positions remain -inf

        # --- 4. softmax (invalid positions -inf -> outputs naturally 0) ---
        o = softmax(x, dim=dim)
        return o

# Main QKV layer function with quantization
def exaq_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        bitwidth: int = 3,
        ) -> torch.Tensor:
    """
      Eager-mode attention with tensor-wise quantized Q/K and int QK^T matmul.
    """

    # Ensure shapes are valid
    assert query.dim() >= 3 and key.dim() >= 3 and value.dim() >= 3, "Q/K/V must be at least 3D"

    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    scale_factor = torch.tensor(scale_factor, device=query.device, dtype=query.dtype)

    if enable_gqa:
        expand_factor = query.size(-3) // key.size(-3)
        key = key.repeat_interleave(expand_factor, dim=-3)
        value = value.repeat_interleave(expand_factor, dim=-3)

    attn_weights = torch.einsum('bhnf,bhmf->bhnm', query, key) * scale_factor  # [B, H, L, S]

    L, S = query.size(-2), key.size(-2)
    attn_bias = torch.zeros_like(attn_weights)

    if is_causal:
        causal_mask = torch.ones((L, S), dtype=torch.bool, device=query.device).tril()
        attn_bias = attn_bias.masked_fill(~causal_mask, float("-inf"))
    else:
        causal_mask = torch.ones((L, S), dtype=torch.bool, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_bias + attn_mask


    attn_weights = attn_weights + attn_bias

    # # Broadcast mask to attn_weights shape
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
    causal_mask = causal_mask.expand_as(attn_weights)  # (B,H,L,S)
    ex = exaq()
    attn_weights = ex.exaq_softmax(
        attn_weights,
        dim=-1,
        mask=causal_mask,
        bitwidth=bitwidth
    )
    attn_weights = attn_weights.to(torch.float16)

    output = torch.einsum('bhnm,bhmf->bhnf', attn_weights, value)
    return output
