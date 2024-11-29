from .BaseAttention import BaseAttention
from SageAttention.sageattention import sageattn

class SageAttention(BaseAttention):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0, quant_bit=None):
        super(SageAttention, self).__init__(embed_dim, num_heads, attn_drop, proj_drop)
        self.quant_max = 2 ** (quant_bit - 1) -1
        self.total_time = 0

    def forward(self, x):
        device = x.device
        B, N, C = x.shape

        # Compute Q, K, V matrices
        qkv = self.qkv(x)  # Linear projection to get concatenated QKV
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # Rearrange dimensions to separate Q, K, V
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape: (B, num_heads, N, head_dim)

        q = q.cuda()
        k = k.cuda()
        v = v.cuda()

        attn_output = sageattn(q, k, v)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        attn_output = attn_output.to(device).float()
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x
