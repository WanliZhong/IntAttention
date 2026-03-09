import torch
import math
from typing import Optional

def tensor_wise_quantize(inp: torch.Tensor, quant_bit: int):
    """
    Perform symmetric tensor-wise quantization to INT8.
    """
    inp_max = torch.max(torch.abs(inp))
    bounds = 1 << (quant_bit - 1)
    factor = inp_max / (bounds - 1)

    int_inp = torch.clamp((inp / factor).round(), -bounds, bounds - 1).to(torch.int8)
    return int_inp, factor

# Main QKV layer function with quantization
def quant_only(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        inp_quant_bit: int = 8,
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

    int_q, factor_q = tensor_wise_quantize(query, inp_quant_bit)  # int8
    int_k, factor_k = tensor_wise_quantize(key, inp_quant_bit)
    int_v, factor_v = tensor_wise_quantize(value, inp_quant_bit)

    int_q = int_q.to(torch.float64)
    int_k = int_k.to(torch.float64)
    int_v = int_v.to(torch.float64)

    int_qk = torch.einsum('bhnf,bhmf->bhnm', int_q, int_k).to(torch.int32)

    L, S = query.size(-2), key.size(-2)

    attn_weights = int_qk.to(torch.float32) * scale_factor * factor_q * factor_k
    attn_bias = torch.zeros_like(attn_weights)

    if is_causal:
        causal_mask = torch.ones((L, S), dtype=torch.bool, device=query.device).tril()
        attn_bias = attn_bias.masked_fill(~causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_bias + attn_mask

    attn_weights = attn_weights + attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = (attn_weights * 127).round().to(torch.float64)

    output = torch.einsum('bhnm,bhmf->bhnf', attn_weights.to(torch.float64), int_v)
    output = (output * factor_v / 127).to(torch.float16)
    return output

