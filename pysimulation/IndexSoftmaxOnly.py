import torch
import math
from typing import Optional, Tuple

def load_table(quant_bit: int, factor: float, device: torch.device | str) -> torch.Tensor:
    """
    Load a precomputed exponential lookup table to approximate softmax.
    """
    num_bins = 1 << quant_bit
    steps = torch.arange(num_bins, device=device)

    # Precompute exponential values and scale them to [0, 255]
    base_table = (torch.exp(steps * -factor / (num_bins - 1)) * 255).round().to(torch.int32)

    # Boundary conditions
    base_table[0] = 255  # Max value for exp(0)
    base_table[-1] = 0   # Bound the lowest value to 0
    return base_table

def tensor_wise_quantize(inp: torch.Tensor, quant_bit: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform symmetric tensor-wise quantization to INT8.
    Note: Retained for utility, though unused in the indexSoftmaxOnly path.
    """
    inp_max = torch.max(torch.abs(inp))
    bounds = 1 << (quant_bit - 1)
    factor = inp_max / (bounds - 1)

    int_inp = torch.clamp((inp / factor).round(), -bounds, bounds - 1).to(torch.int8)
    return int_inp, factor

def idx_softmax_fp(inp: torch.Tensor,
                   quant_bit: int,
                   default_zero_thr: float) -> torch.Tensor:
    """
    Softmax computed using index-based quantization and lookup tables for float inputs.
    """
    # 1. Shift for numerical stability (distances from max)
    row_max = torch.max(inp, dim=-1, keepdim=True).values
    _inp = row_max - inp  # Positive values representing distance from max

    # 2. Apply threshold clamping directly after subtraction
    _inp = torch.clamp(_inp, max=default_zero_thr)

    # 3. Scale to lookup table index range
    num_bins = (1 << quant_bit) - 1
    _inp = ((_inp * num_bins) / default_zero_thr).to(torch.int32)

    # 4. Load base precomputed exponentials
    base_table = load_table(quant_bit, default_zero_thr, inp.device)

    # 5. Use tensor indexing for exponentials
    exp_res = base_table[_inp]

    # 6. Normalize (compute softmax)
    sum_exp_res = torch.sum(exp_res, dim=-1, keepdim=True)

    # Add epsilon to prevent division by zero
    softmax_out = exp_res / sum_exp_res

    return softmax_out

def idx_softmax_only(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        inp_quant_bit: int = 8,  # Kept for signature compatibility
        quant_bit: int = 5,
        zero_thr: float = 6.6) -> torch.Tensor:
    """
    Eager-mode attention using standard FP32/FP16 QK^T matmul, 
    but with an index-based quantized Softmax approximation.
    """
    # Ensure shapes are valid
    assert query.dim() >= 3 and key.dim() >= 3 and value.dim() >= 3, "Q/K/V must be at least 3D"

    # Handle scaling
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale

    # Handle Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
    if enable_gqa:
        expand_factor = query.size(-3) // key.size(-3)
        key = key.repeat_interleave(expand_factor, dim=-3)
        value = value.repeat_interleave(expand_factor, dim=-3)

    # 1. Standard Q * K^T Matmul -> Shape: [B, H, L, S]
    attn_weights = torch.einsum('bhnf,bhmf->bhnm', query, key) * scale_factor

    L, S = query.size(-2), key.size(-2)

    # 2. Apply Masks directly to attn_weights (Memory Efficient)
    if is_causal:
        causal_mask = torch.ones((L, S), dtype=torch.bool, device=query.device).tril()
        attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attn_mask

    # 3. Approximate Softmax using Index Lookup
    attn_weights = idx_softmax_fp(attn_weights, quant_bit, zero_thr).to(torch.float16)

    # 4. Output Projection (Attn_weights * V)
    output = torch.einsum('bhnm,bhmf->bhnf', attn_weights, value)

    return output