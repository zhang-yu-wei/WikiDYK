#!/bin/bash

# Language Model Evaluation Script for Multiple Models
# This script runs the language model evaluation on WikiDYK data for multiple models
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Input file and common parameters
INPUT_FILE="data/wikidyk2022-2025_01082025_gpt-4o_evalv2_pages_formatted_combined_v2.json"
OUTPUT_DIR="eval_results_v2"
# Infer tensor parallel size and visible devices
TENSOR_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE + 1))
GPU_MEMORY_UTIL=0.95
MAX_NEW_TOKENS=256
MODEL_MAX_LEN=32768
RAG_TOP_K=3
USE_CHAT_MODE=false

# Models to evaluate
MODELS=(
#   "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Llama-3.1-8B"
  "Qwen/Qwen2.5-1.5B"
  "Qwen/Qwen2.5-7B"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation for each model
for MODEL_NAME in "${MODELS[@]}"; do
  echo "---------------------------------------------"
  echo "Evaluating model: $MODEL_NAME"
  echo "Started at: $(date)"

  if [[ "$MODEL_NAME" == *"Llama-2"* ]]; then
    MODEL_MAX_LEN=4096
  fi
  
  # Build command
  CMD="python src/eval_qa.py --model_name \"$MODEL_NAME\" --input_file \"$INPUT_FILE\" \
      --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
      --gpu_memory_utilization $GPU_MEMORY_UTIL \
      --max_new_tokens $MAX_NEW_TOKENS \
      --model_max_len $MODEL_MAX_LEN"
  
  # Add chat mode for all models
  if [ "$USE_CHAT_MODE" = true ]; then
    CMD="$CMD --chat_mode"
  fi
  
  # Add quantization for large models
  if [[ "$MODEL_NAME" == *"70B"* ]]; then
    CMD="$CMD --quantization"
  fi

  # Add RAG top-k for RAG models
  if [[ "$RAG_TOP_K" -gt 0 ]]; then
    CMD="$CMD --eval_rag --rag_top_k $RAG_TOP_K"
  fi
  
  # Print and execute command
  echo "Command: $CMD"
  eval "$CMD"
  echo "Finished at: $(date)"
  echo "---------------------------------------------"
done

echo "All evaluations completed at $(date)"
