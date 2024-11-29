import torch
import time
from pysrc.IndexAttention import IndexAttention


# 定义测试函数
def benchmark_softmax(batch_size, seq_len, head_dim, num_heads, iterations=10000):
    # 初始化随机输入张量
    inp = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device="cuda")

    # 初始化 IndexAttention
    index_attention = IndexAttention(embed_dim=head_dim * num_heads, num_heads=num_heads, quant_bit=4, zero_thr=5).cuda()

    # torch.softmax 测试
    torch_softmax_times = []
    torch.cuda.synchronize()
    for _ in range(iterations):
        start = time.time()
        _ = inp.softmax(dim=-1)
        torch.cuda.synchronize()
        torch_softmax_times.append(time.time() - start)
    torch_softmax_mean_time = sum(torch_softmax_times) / iterations
    torch_softmax_min_time = min(torch_softmax_times)

    # idxSoftmax 测试
    idx_softmax_times = []
    torch.cuda.synchronize()
    for _ in range(iterations):
        start = time.time()
        _ = index_attention.idxSoftmax(inp, default_zero_thr=10, dim=-1)
        torch.cuda.synchronize()
        idx_softmax_times.append(time.time() - start)
    idx_softmax_mean_time = sum(idx_softmax_times) / iterations
    idx_softmax_min_time = min(idx_softmax_times)

    return torch_softmax_mean_time, torch_softmax_min_time, idx_softmax_mean_time, idx_softmax_min_time


# 配置测试参数
batch_size = 1  # 增大批量大小
seq_len = 197    # 增大序列长度
head_dim = 197
num_heads = 12
iterations = 10000  # 测试 10000 次

# 执行测试
torch_softmax_mean_time, torch_softmax_min_time, idx_softmax_mean_time, idx_softmax_min_time = benchmark_softmax(
    batch_size, seq_len, head_dim, num_heads, iterations
)

# 打印测试结果
print(f"torch.softmax average time: {torch_softmax_mean_time * 1e6:.2f} μs")
print(f"torch.softmax minimum time: {torch_softmax_min_time * 1e6:.2f} μs")
print(f"idxSoftmax average time: {idx_softmax_mean_time * 1e6:.2f} μs")
print(f"idxSoftmax minimum time: {idx_softmax_min_time * 1e6:.2f} μs")