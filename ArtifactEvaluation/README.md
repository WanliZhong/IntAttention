# Artifact Evaluation

This directory contains the code, data, and instructions needed to reproduce the main artifact results for **IntAttention: A Fully Integer Attention Pipeline for Efficient Edge Inference**.

This README is the authoritative artifact evaluation workflow for the MLSys 2026 submission.

## Supported Claims

1. **Accuracy:** IntAttention preserves high-fidelity accuracy relative to FP16 and baseline attention across both LLMs and Vision Transformers. This corresponds to **Table 1** and **Table 2** in the paper.
2. **Latency:** IntAttention reduces attention latency on Arm CPUs relative to FP32, FP16, and INT8 quantization baselines. This corresponds to **Figure 6** and **Figure 7** in the paper.

## Requirements

The artifact has two separate execution paths because the latency and accuracy experiments require different environments.

### 1. Latency Evaluation

- **Hardware:** Armv8.6-a CPU. The reference configuration targets Apple M-series chips, but other Armv8.6-a platforms can also be used.
- **Software:** `scons`, `clang`

### 2. Accuracy Evaluation

- **Hardware:** CUDA-capable GPU with at least 10 GB VRAM
- **Software:** Python 3.10+, `pip`

Note: native INT32 matrix multiplication support is limited on current GPUs, so the PyTorch simulation uses high-precision arithmetic to emulate the intended integer attention behavior for accuracy validation.

## Part 1: Latency Evaluation on Arm CPU

### 1. Clone the Repository

From the repository root:

```bash
git clone --recursive https://github.com/WanliZhong/IntAttention
cd IntAttention
cd ArtifactEvaluation
```

If you pulled after the repository restructure, run this once from the repository root before entering `ArtifactEvaluation/`:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### 2. Prepare ARM ComputeLibrary

```bash
cd ComputeLibrary
git checkout v52.5.0
git apply ../add_impl_for_ACL.patch
```

### 3. Build ComputeLibrary

Reference build command:

```bash
scons -j"$(sysctl -n hw.ncpu)" \
  os=macos arch=armv8.6-a \
  neon=1 opencl=0 embed_kernels=0 logging=0 \
  Werror=0 debug=0 asserts=0 examples=0 \
  extra_cxx_flags="-mcpu=apple-m2"
```

### Platform Notes

The command above is the reference configuration used for the artifact. On other Armv8.6-a platforms, small adjustments may be required.

- **Apple Silicon:** keep `os=macos` and replace `-mcpu=apple-m2` with the closest target for your chip if needed.
- **Other Arm Linux boards:** you will likely need to change `os=linux` and adapt `extra_cxx_flags` to the target CPU.
- **General guidance:** the important requirement is Armv8.6-a plus NEON support. If your machine differs from the reference setup, use the closest matching target rather than copying the flags verbatim.

### 4. Build the Benchmark

```bash
cd ..
INCDIR="./ComputeLibrary"
LIBDIR="./ComputeLibrary/build"

clang++ bench_speed.cpp -O3 -std=c++17 -arch arm64 \
  -I "$INCDIR/include" -I "$INCDIR" \
  "$LIBDIR/libarm_compute-static.a" \
  "$LIBDIR/libarm_compute_graph-static.a" \
  -o bench_speed -lpthread -ldl
```

### 5. Run the Benchmark

The benchmark supports four pipelines:

- `--pipe 0`: FP32
- `--pipe 1`: FP16
- `--pipe 2`: Quant-Only INT8 baseline
- `--pipe 3`: IntAttention

Example:

```bash
# IntAttention
./bench_speed --pipe 3 --L 1024 --d 128 --warmup 10 --runs 100

# FP16 baseline
./bench_speed --pipe 1 --L 1024 --d 128 --warmup 10 --runs 100
```

Here `L` is the sequence length and `d` is the head dimension.

### Energy Measurement Data

The joule-level energy numbers in the paper were measured on a **Khadas Edge2** device using an external **POWER-Z KM003C** USB power meter.

These energy measurements are not produced directly by `bench_speed.cpp`. Instead, the benchmark was executed on the target device while the external meter logged the power trace over time.

The raw trace files are stored in `power_traces/`. Their naming convention is:

- `fp32`, `fp16`, `quant_only`, `int_attention`: the four benchmark pipelines, corresponding to `--pipe 0`, `1`, `2`, and `3`
- `L1k`, `L2k`, `L4k`, `L8k`, `L16k`: sequence length
- `repeatXXX`: repeated-run setting used during that power capture

Examples:

- `power_traces/fp16_L4k_repeat400.csv`
- `power_traces/int_attention_L8k_repeat200.csv`

Each CSV is a raw export from the external power meter and includes elapsed time, voltage, current, accumulated energy, and instantaneous power. The energy numbers reported in the paper were obtained by selecting a stable execution interval from these traces and aggregating that interval offline.

If you run the artifact without the same external measurement setup, you should expect to reproduce the **latency** trends, but not the exact **joule** values from the paper.

## Part 2: Accuracy Evaluation on GPU

### 1. Install Dependencies

Run the following commands from the `ArtifactEvaluation/` directory:

```bash
pip install torch==2.8.0 torchvision==0.23.0 transformers==4.57.0 timm==1.0.19 lm_eval==0.4.9.2
pip install langdetect immutabledict
export HF_ALLOW_CODE_EVAL=1
```

### 2. Language Models

We support LLM evaluation through `acc_llm.py`. Example models include:

- `meta-llama/Llama-3.2-1B`
- `facebook/opt-1.3b`
- `Qwen/Qwen3-1.7B`

Example:

```bash
python acc_llm.py --model-name meta-llama/Llama-3.2-1B \
  --method int_attention \
  --tasks wikitext hellaswag lambada_openai piqa winogrande arc_challenge arc_easy
```

Small numerical differences across hardware and software environments are possible, but the method ranking and overall trends should remain consistent with the paper.

### 3. Vision Transformers

For ViT evaluation, ensure that the ImageNet-1k validation set is available locally.

`acc_deit.py` is a compact ImageNet evaluation entrypoint. It reuses the bundled DeiT evaluation components, infers the required input resolution from the model name, and injects the selected attention backend at runtime. No source patching is required.

Example:

```bash
python acc_deit.py --model deit_base_patch16_224 \
  --data-path path/to/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/ \
  --method int_attention
```

As with the LLM path, small numerical differences are possible across environments, but the ranking and overall trends should match the paper.
