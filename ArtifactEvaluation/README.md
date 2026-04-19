# Artifact Evaluation

This directory contains the code, data, and instructions needed to reproduce the main artifact results for IntAttention.

The workflow documented in this README is the authoritative artifact evaluation path for the MLSys 2026 submission.

## Supported Claims

1. **Claim 1 (Accuracy):** Our fully integer pipeline (IntAttention) maintains high-fidelity accuracy relative to FP16 and baseline attention across various LLMs and Vision Transformers. *(Corresponds to Table 1 and Table 2 in the paper).*
2. **Claim 2 (Latency/Speed):** The proposed IntAttention pipeline achieves significant latency reduction on Arm CPUs compared to FP32, FP16, and INT8 quantization baselines. *(Corresponds to Figure 6 and Figure 7 in the paper).*


## Directory Structure

* `pysimulation/`: PyTorch-based simulation of IntAttention and the baseline attention variants.
* `add_impl_for_ACL.patch`: Patch for integrating the IndexSoftmax operator into the ARM ComputeLibrary.
* `acc_llm.py`: Evaluation script for Large Language Models.
* `acc_deit.py`: Evaluation script for Vision Transformers.
* `bench_speed.cpp`: C++ benchmarking script for evaluating latency on Armv8 CPUs.
* `power_traces/`: Raw power traces used to prepare the energy plots reported in the paper.
* `ComputeLibrary/`: submodule used for latency benchmarking.
* `deit/`: submodule reused by `acc_deit.py` for ViT evaluation.


## Requirements

Due to the fundamental differences in hardware execution, our evaluation is split into two environments:

### 1. Latency / Speed Evaluation

* **Hardware:** Armv8.6-a architecture device. The reference build configuration targets Apple M-series chips (e.g., M2/M3/M4), but can be adapted to other Armv8.6-a platforms.
* **Software:** `scons`, `clang`.

### 2. Accuracy Evaluation

* **Hardware:** CUDA-compatible GPU with at least 10GB VRAM. *(Note: Because native INT32 matrix multiplication on GPUs is limited, our PyTorch simulation uses FP64 to guarantee precise INT32 emulation for accuracy validation).*
* **Software:** Python 3.10+, `pip`.


## 🚀 Part 1: Latency Evaluation (ARM CPU)

### 1. Setup and Build ComputeLibrary

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

Checkout the required version of ComputeLibrary and apply the IndexSoftmax patch:

```bash
cd ComputeLibrary
git checkout v52.5.0
git apply ../add_impl_for_ACL.patch
```

Compile the ComputeLibrary:

```bash
scons -j"$(sysctl -n hw.ncpu)" \
  os=macos arch=armv8.6-a \
  neon=1 opencl=0 embed_kernels=0 logging=0 \
  Werror=0 debug=0 asserts=0 examples=0 \
  extra_cxx_flags="-mcpu=apple-m2"
```

### Platform Notes

The `scons` command above is the reference configuration used for the artifact. On other Armv8.6-a platforms, small platform-specific adjustments may be needed:

* **Apple Silicon (M2/M3/M4):** keep `os=macos` and replace `-mcpu=apple-m2` with the closest available target for your chip if needed.
* **Other Arm Linux boards:** you will likely need to change `os=linux` and adjust `extra_cxx_flags` to match the target CPU.
* **General guidance:** the key requirement for the latency path is an Armv8.6-a compatible CPU with NEON support. If your platform differs from the reference setup, use the closest matching `scons` target rather than copying the exact flags verbatim.

### 2. Compile and Run Benchmark

Compile the C++ benchmark script:

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

Run the speed tests. The benchmark supports 4 pipelines (`--pipe 0` to `3`):

* `0` (Pure FP32): QK(FP32) -> Softmax(FP32) -> PV(FP32)
* `1` (FP16): QK(F16) -> Cast(F32) -> Softmax(F32) -> Cast(F16) -> PV(F16)
* `2` (Quantized INT8): S8 QK -> S32 -> FP32 Softmax -> S8 PV -> S32 -> F16
* `3` (**IntAttention**): S8 QK -> S32 -> IndexSoftmax(U8) -> U8xS8 PV -> S32 -> F16

**Example Command (IntAttention vs FP16):**

```bash
# Run IntAttention
./bench_speed --pipe 3 --L 1024 --d 128 --warmup 10 --runs 100

# Run FP16 Baseline for comparison
./bench_speed --pipe 1 --L 1024 --d 128 --warmup 10 --runs 100
```

*(Where `L` is the sequence length and `d` is the head dimension).*

### Energy Measurement Data

The joule-level energy numbers in the paper were measured on a **Khadas Edge2** device using an external **POWER-Z KM003C** USB power meter. These measurements are not reproduced directly by the benchmarking code above; instead, the benchmark was executed on the target device while the external meter logged power over time.

The raw trace files are provided under `power_traces/`. Their naming convention is:

* `fp32`, `fp16`, `quant_only`, `int_attention`: the four benchmark pipelines, corresponding to `--pipe 0`, `1`, `2`, and `3`
* `L1k`, `L2k`, `L4k`, `L8k`, `L16k`: sequence length
* `repeatXXX` when present: the repeated-run setting used during the power capture

For example:

* `power_traces/fp16_L4k_repeat400.csv`
* `power_traces/int_attention_L8k_repeat200.csv`

Each CSV is a raw export from the external power meter and includes columns such as elapsed time, voltage, current, accumulated energy, and instantaneous power. The energy figures in the paper were obtained by selecting a stable execution interval from these traces and then aggregating that interval offline.

If you are running the artifact on a generic machine without the same external measurement setup, you should expect to reproduce the **latency** trends but not the exact **joule** values from the paper.


## 🎯 Part 2: Accuracy Evaluation (GPU)

### 1. Install Dependencies

Run the following commands from the `ArtifactEvaluation/` directory:

```bash
pip install torch==2.8.0 torchvision==0.23.0 transformers==4.57.0 timm==1.0.19 lm_eval==0.4.9.2
pip install langdetect immutabledict
export HF_ALLOW_CODE_EVAL=1
```

### 2. Language Models (LLMs) Evaluation

We support evaluating `meta-llama/Llama-3.2-1B`, `facebook/opt-1.3b`, and `Qwen/Qwen3-1.7B` on standard zero-shot tasks.

As with the original artifact evaluation, small numerical differences across hardware and software environments are possible, but the method ranking and overall trends should remain consistent with the paper.

**Example Command (Llama-3.2-1B with IntAttention):**

```bash
python acc_llm.py --model-name meta-llama/Llama-3.2-1B \
  --method int_attention \
  --tasks wikitext hellaswag lambada_openai piqa winogrande arc_challenge arc_easy
```

### 3. Vision Models (ViT) Evaluation

Ensure you have the `ImageNet-1k` validation dataset downloaded. We support models including `deit_base_patch16_224`, `vit_large_patch16_384`, and `cait_large_patch16_448`.

`acc_deit.py` is a compact evaluation entrypoint for ImageNet-1k validation. It reuses the bundled DeiT evaluation components, infers the required input resolution from the model name, and injects the selected attention backend at runtime, so no source patching is required.

As with the LLM path, small numerical differences across hardware and software environments are possible, but the method ranking and overall trends should remain consistent with the paper.

**Example Command (DeiT-B-224 with IntAttention):**

```bash
python acc_deit.py --model deit_base_patch16_224 \
  --data-path path/to/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/ \
  --method int_attention
```
