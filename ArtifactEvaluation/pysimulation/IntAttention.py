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
    """
    inp_max = torch.max(torch.abs(inp))
    bounds = 1 << (quant_bit - 1)
    factor = inp_max / (bounds - 1)

    int_inp = torch.clamp((inp / factor).round(), -bounds, bounds - 1).to(torch.int8)
    return int_inp, factor

def idx_softmax(inp: torch.Tensor,
                factor_q: torch.Tensor,
                factor_k: torch.Tensor,
                scale: torch.Tensor,
                mask: torch.Tensor,
                quant_bit: int,
                default_zero_thr: float) -> torch.Tensor:
    """
    Softmax computed using index-based quantization and lookup tables.
    """
    # 1. Masking: Fill invalid positions with a large negative number
    # Assuming mask is True for valid tokens and False for padding/future tokens
    inp = inp.masked_fill(~mask, -1048576)

    # 2. Shift for numerical stability (distances from max)
    row_max = torch.max(inp, dim=-1, keepdim=True).values
    _inp = row_max - inp  # Positive values representing distance from max

    # 3. Apply threshold clamping
    zero_thr = (((default_zero_thr / factor_q) / factor_k) / scale).to(torch.int32)
    _inp = torch.min(_inp, zero_thr)

    # 4. Scale to lookup table index range
    num_bins = (1 << quant_bit) - 1
    _inp = ((_inp * num_bins) / zero_thr).to(torch.int32)

    # 5. Load table and fetch exponentials
    base_table = load_table(quant_bit, default_zero_thr, inp.device)
    exp_res = base_table[_inp]

    # 6. Apply mask to exponentials (pad tokens contribute 0 to the sum)
    exp_res = exp_res.masked_fill(~mask, 0)

    # 7. Normalize (compute softmax)
    sum_exp_res = torch.sum(exp_res, dim=-1, keepdim=True)
    # Scale to 255 and add epsilon to prevent division by zero
    softmax_out = ((exp_res * 255.0) / sum_exp_res).round().to(torch.int32)

    return softmax_out

def int_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        inp_quant_bit: int = 8,
        quant_bit: int = 5,
        zero_thr: float = 6.6) -> torch.Tensor:
    """
    Eager-mode attention with tensor-wise quantized Q/K/V and integer QK^T matmul.
    Shapes expected: (Batch, Heads, SeqLen, HeadDim)
    """
    assert query.dim() >= 3 and key.dim() >= 3 and value.dim() >= 3, "Q/K/V must be at least 3D"

    # Default scale is 1 / sqrt(d_k)
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    scale_factor = torch.tensor(scale_factor, device=query.device, dtype=torch.float32)

    # Handle Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
    if enable_gqa:
        expand_factor = query.size(-3) // key.size(-3)
        key = key.repeat_interleave(expand_factor, dim=-3)
        value = value.repeat_interleave(expand_factor, dim=-3)

    # 1. Quantize Inputs
    int_q, factor_q = tensor_wise_quantize(query, inp_quant_bit)
    int_k, factor_k = tensor_wise_quantize(key, inp_quant_bit)
    int_v, factor_v = tensor_wise_quantize(value, inp_quant_bit)

    # 2. Simulate INT32 Matmul using FP64
    # FP64 can perfectly represent exact integers up to 2^53.
    int_q = int_q.to(torch.float64)
    int_k = int_k.to(torch.float64)
    int_v = int_v.to(torch.float64)

    # Q * K^T -> Shape: [B, H, N, S]
    int_qk = torch.einsum('bhnf,bhmf->bhnm', int_q, int_k).to(torch.int32)

    L, S = query.size(-2), key.size(-2)

    # 3. Create Attention Mask
    # Initial mask: All True (all positions visible)
    final_mask = torch.ones((L, S), dtype=torch.bool, device=int_qk.device)

    # If causal, keep only the lower triangular portion
    if is_causal:
        final_mask = final_mask.tril()

    # Expand dimensions to (1, 1, L, S) to rely on broadcasting for (B, H, L, S)
    final_mask = final_mask.unsqueeze(0).unsqueeze(1)

    # Combine with provided attention mask if any
    if attn_mask is not None:
        final_mask = final_mask & attn_mask

    # 4. Compute Integer Softmax
    attn_weights = idx_softmax(
        inp=int_qk,
        factor_q=factor_q,
        factor_k=factor_k,
        scale=scale_factor,
        mask=final_mask,
        quant_bit=quant_bit,
        default_zero_thr=zero_thr
    )

    # 5. Output Projection (Attn_weights * V)
    output = torch.einsum('bhnm,bhmf->bhnf', attn_weights.to(torch.float64), int_v)

    # De-quantize back to float16 scale
    output = ((output * factor_v) / 255.0).to(torch.float16)

    return output

