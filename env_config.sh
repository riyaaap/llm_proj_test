#!/bin/bash
# 1. Hardware Boundary Rule
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 2. Structural Fix: Shut down the V1 Engine Core to force stable V0 execution fallback
export VLLM_USE_V1=0

# 3. FlashInfer JIT Bypasses
export VLLM_DISABLE_FLASHINFER=1
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export FLASHINFER_DISABLE_JIT=1


