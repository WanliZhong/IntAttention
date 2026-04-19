# IntAttention: A Fully Integer Attention Pipeline for Efficient Edge Inference

[[Paper](https://arxiv.org/abs/2511.21513)] [[MLSys](https://mlsys.org/virtual/2026/oral/3848)] [[Artifact Evaluation](./ArtifactEvaluation/README.md)]

This repository contains the official code for **"IntAttention: A Fully Integer Attention Pipeline for Efficient Edge Inference"** (MLSys 2026).

The artifact evaluation workflow for reproducing the paper results is documented in [ArtifactEvaluation/README.md](./ArtifactEvaluation/README.md).

![](./assets/pipeline.jpg)

## Overview

IntAttention is a fully integer attention pipeline designed for efficient edge inference. Instead of falling back to floating-point softmax and value mixing after integer QK accumulation, IntAttention keeps the whole attention path in low precision:

- `S8 x S8 -> S32` for query-key accumulation
- `S32 -> U8` IndexSoftmax for probability generation
- `U8 x S8 -> S32` for probability-value mixing

Compared with conventional INT8 attention pipelines that dequantize to floating point around softmax, IntAttention preserves an integer computation path throughout attention, reducing memory traffic and improving CPU efficiency while maintaining accuracy.

