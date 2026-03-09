import torch
import argparse
from lm_eval import evaluator  # EleutherAI lm-evaluation-harness
from lm_eval.models.huggingface import HFLM  # Hugging Face adapter
import torch.nn.functional as F
from functools import partial
import pysimulation

def configure_intattention(method_name="IntAttention", inp_quant_bit=8, quant_bit=5, zero_thr=6.6, bitwidth=3):
    """Configure the scaled dot-product attention implementation.

    Supported method names (case-insensitive):
      - int_attention / IntAttention
      - idx_softmax_only / idx_softmax
      - exaq_attention / exaq
      - quant_only / quant
      - (default) None: leave default PyTorch implementation

    This maps the chosen implementation from `pysimulation` into
    `torch.nn.functional.scaled_dot_product_attention` using `functools.partial`.
    Extra keyword parameters specific to implementations are passed through.
    """
    if not method_name:
        return

    m = method_name.lower()

    if m in ("int_attention", "intattention", "int-attention"):
        F.scaled_dot_product_attention = partial(
            pysimulation.int_attention,
            inp_quant_bit=inp_quant_bit,
            quant_bit=quant_bit,
            zero_thr=zero_thr,
        )
    elif m in ("idx_softmax_only", "idx_softmax", "idxsoftmaxonly"):
        # idx_softmax_only expects (query,key,value, ... , quant_bit, zero_thr)
        F.scaled_dot_product_attention = partial(
            pysimulation.idx_softmax_only,
            inp_quant_bit=inp_quant_bit,
            quant_bit=quant_bit,
            zero_thr=zero_thr,
        )
    elif m in ("exaq_attention", "exaq", "exaq_attention"):
        F.scaled_dot_product_attention = partial(
            pysimulation.exaq_attention,
            bitwidth=bitwidth,
        )
    elif m in ("quant_only", "quantonly", "quant_only"):
        F.scaled_dot_product_attention = partial(
            pysimulation.quant_only,
            inp_quant_bit=inp_quant_bit,
        )
    else:
        # Unknown method: leave default PyTorch implementation
        print(f"[WARN] Unknown attention method '{method_name}', leaving default attention.")

def evaluate_model(model_name, batch_size=1, tasks=None, dtype=torch.float16, max_length=4096):
    """Evaluate model accuracy using a small, quick dataset by default.

    `tasks` can be a list of lm-eval task names.
    """
    if tasks is None:
        tasks = ["piqa"]
    # Examples: ['paloma', 'c4', 'humaneval'] — PIQA is small and fast
    # Other options: ['humaneval', 'mbpp', 'gsm8k', 'ifeval']
    # Or variants: ['paloma_c4_en', 'paloma_redpajama']
    hf_model = HFLM(
        pretrained=model_name,
        batch_size=batch_size,
        dtype=dtype,
        low_cpu_mem_usage=True,
        max_length=max_length,
        trust_remote_code=True,
    )

    eval_results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=tasks,
        confirm_run_unsafe_code=True
    )
    return eval_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a HuggingFace causal LM with optional IntAttention integration")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B", help="pretrained model name or path")
    parser.add_argument("--method", default=None, help="method for attention (int_attention, idx_softmax_only, exaq_attention, quant_only)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16","float32"], default="float16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--tasks", nargs="+", default=["piqa"], help="one or more tasks for lm-eval (e.g. --tasks piqa gsm8k)")
    parser.add_argument("--inp-quant-bit", type=int, default=8)
    parser.add_argument("--quant-bit", type=int, default=5)
    parser.add_argument("--zero-thr", type=float, default=6.6)
    parser.add_argument("--bitwidth", type=int, default=3, help="bitwidth for EXAQ (default: 3)")
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    METHOD_NAME = args.method

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using {METHOD_NAME} for scaled dot product attention")

    # configure IntAttention implementation
    configure_intattention(method_name=METHOD_NAME, inp_quant_bit=args.inp_quant_bit, quant_bit=args.quant_bit, zero_thr=args.zero_thr, bitwidth=args.bitwidth)

    print("Evaluating model...")
    dtype_obj = torch.float16 if args.dtype == "float16" else torch.float32
    tasks_list = args.tasks
    eval_results = evaluate_model(MODEL_NAME, batch_size=args.batch_size, tasks=tasks_list, dtype=dtype_obj, max_length=args.max_length)

    print("\n=== Evaluation results ===")
    print(f"Using {METHOD_NAME} for scaled dot product attention")
    for task in eval_results.get("results", {}):
        print(f"{task}")
        for item in eval_results["results"][task]:
            print(f"{item}: {eval_results['results'][task][item]}")

if __name__ == "__main__":
    main()