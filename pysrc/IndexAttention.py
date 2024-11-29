import torch
from .BaseAttention import BaseAttention
from .idxsoftmax import idxsoftmax

class IndexAttention(BaseAttention):
    base_cache = None
    inv_scale_factor = None

    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0, quant_bit=8, zero_thr=10):
        super(IndexAttention, self).__init__(embed_dim, num_heads, attn_drop, proj_drop)
        self.quant_bit = quant_bit
        self.zero_thr = zero_thr
        if IndexAttention.base_cache is None:
            IndexAttention.base_cache = [None] * zero_thr
        if IndexAttention.inv_scale_factor is None:
            IndexAttention.inv_scale_factor = [None] * zero_thr

    def load_base(self, zero_thr, device):
        if IndexAttention.base_cache[zero_thr - 1] is None:
            inv_scale_factor = (2 ** self.quant_bit - 1) / zero_thr
            base = torch.exp(torch.arange(2 ** self.quant_bit, dtype=torch.float16) / -inv_scale_factor)
            base[-1] = 0
            IndexAttention.base_cache[zero_thr - 1] = base
            IndexAttention.inv_scale_factor[zero_thr - 1] = inv_scale_factor
        base = IndexAttention.base_cache[zero_thr - 1].to(device)
        return base, IndexAttention.inv_scale_factor[zero_thr - 1]

    def forward(self, x):
        device = x.device
        B, N, C = x.shape

        # Compute Q, K, V matrices
        qkv = self.qkv(x)  # Linear projection to get concatenated QKV
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # Rearrange dimensions to separate Q, K, V
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, B, num_heads, N, head_dim)
        qkv = qkv.half()
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape: (B, num_heads, N, head_dim)

        q = q.to(device)
        k = k.to(device)
        v = v.to(device)

        attn_output = self.idxattn(q, k, v)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        attn_output = attn_output.to(device).float()
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x

    def idxattn(self, q, k, v):
        B, num_heads, N, head_dim = q.shape

        q = q * (head_dim ** -0.5)

        attn_scores = torch.einsum('bhnf,bhmf->bhnm', q, k)  # Shape: (B, num_heads, N, N)

        # attn_weights = attn_scores.softmax(dim=-1)  # Shape: (B, num_heads, N, N)
        attn_weights = self.idxSoftmax(attn_scores, self.zero_thr, dim=-1)
        # base, inv_scale = self.load_base(self.zero_thr, q.device)
        # attn_weights = idxsoftmax(attn_scores, base, self.zero_thr, inv_scale)

        if self.attn_drop is not None:
            attn_weights = self.attn_drop(attn_weights)

        attn_output = torch.einsum('bhnm,bhmf->bhnf', attn_weights, v)  # Shape: (B, num_heads, N, head_dim)

        return attn_output

    def idxSoftmax(self, inp, default_zero_thr, dim=-1):
        # Step 1: Subtract the max value along the specified dimension for numerical stability
        row_max = inp.max(dim=dim, keepdim=True).values
        global_max = row_max.max()
        zero_thr = min(int(global_max) + 1, default_zero_thr)
        base, inv_scale = self.load_base(zero_thr, inp.device)
        quantized_inp = torch.clamp(row_max - inp, min=0, max=zero_thr)

        # Step 2: Quantization
        quantized_inp = (quantized_inp * inv_scale).round().int().contiguous()
        # Step 3: Cached lookup for exponentials
        exp_res = base[quantized_inp]

        # Step 4: Compute softmax
        sum_exp = 1 /  exp_res.sum(dim=dim, keepdim=True, dtype=torch.float16)
        softmax_out = exp_res * sum_exp

        return softmax_out


