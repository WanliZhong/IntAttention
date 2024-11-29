import torch
from torch.autograd import Function
# import idxsoftmax_cuda

class IdxSoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, inp, base, zero_thr, inv_scale):
        out = torch.empty_like(inp)
        # idxsoftmax_cuda.idxsoftmax_forward(inp.contiguous(), out, base, zero_thr, inv_scale)
        return out

def idxsoftmax(inp, base, zero_thr, inv_scale):
    return IdxSoftmaxFunction.apply(inp, base, zero_thr, inv_scale)
