from functools import partial

import torch.nn.functional as F

from .EXAQ import exaq_attention as _exaq_attention
from .IndexSoftmaxOnly import idx_softmax_only as _idx_softmax_only
from .IntAttention import int_attention as _int_attention
from .QuantOnly import quant_only as _quant_only


def configure_attention(method_name=None, inp_quant_bit=8, quant_bit=5, zero_thr=6.6, bitwidth=3):
    """Install an alternate scaled dot-product attention backend."""
    if not method_name:
        print("[INFO] No attention method specified, leaving default attention.")
        return

    method = method_name.lower()

    if method in ("int_attention", "intattention", "int-attention"):
        F.scaled_dot_product_attention = partial(
            _int_attention,
            inp_quant_bit=inp_quant_bit,
            quant_bit=quant_bit,
            zero_thr=zero_thr,
        )
    elif method in ("idx_softmax_only", "idx_softmax", "idxsoftmaxonly"):
        F.scaled_dot_product_attention = partial(
            _idx_softmax_only,
            inp_quant_bit=inp_quant_bit,
            quant_bit=quant_bit,
            zero_thr=zero_thr,
        )
    elif method in ("exaq_attention", "exaq"):
        F.scaled_dot_product_attention = partial(
            _exaq_attention,
            bitwidth=bitwidth,
        )
    elif method in ("quant_only", "quantonly"):
        F.scaled_dot_product_attention = partial(
            _quant_only,
            inp_quant_bit=inp_quant_bit,
        )
    else:
        print(f"[WARN] Unknown attention method '{method_name}', leaving default attention.")


__all__ = ["configure_attention"]
