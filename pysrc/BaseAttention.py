from torch import nn

class BaseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(BaseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented in BaseAttention class")
