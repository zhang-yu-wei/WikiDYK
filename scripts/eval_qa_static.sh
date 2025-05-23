#!/bin/bash

# Language Model Evaluation Script for Multiple Models
# This script runs the language model evaluation on WikiDYK data for multiple models
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Input file and common parameters
INPUT_FILE="data/wikidyk2022-2025_01082025_gpt-4o_evalv2_pages_formatted_combined_v2.json"
OUTPUT_DIR="eval_results"
# Infer tensor parallel size and visible devices
TENSOR_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE + 1))
GPU_MEMORY_UTIL=0.90
MAX_NEW_TOKENS=256
MODEL_MAX_LEN=1024
RAG_TOP_K=0
USE_CHAT_MODE=false
PEFT=false
# Use base model name for LoRA
BASE_MODEL_NAME=""
# ===== Modify the following parameters as needed =====
DS_SIZE=-1
OVERWRITE=false

# Models to evaluate
MODELS=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B"
    "Qwen/Qwen-2.5-1.5B"
    "Qwen/Qwen-2.5-7B"
    "google/gemma-3-1b-pt"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to extract model size in billions
get_model_size() {
    local model_name=$1
    
    # Extract the size suffix (like 7b, 3B, 14B, etc.)
    if [[ $model_name =~ ([0-9]+\.?[0-9]*)([Bb]) ]]; then
        local size=${BASH_REMATCH[1]}
        echo $size
    else
        echo 0
    fi
}

# Run evaluation for each model
for MODEL_NAME in "${MODELS[@]}"; do
  echo "---------------------------------------------"
  echo "Evaluating model: $MODEL_NAME"
  echo "Started at: $(date)"

  # infer predict mask based on model name
  if [[ $MODEL_NAME == *"predict_mask"* ]]; then
    PREDICT_MASK=true
  else
    PREDICT_MASK=false
  fi

  # Check if model size is over 3B to use LoRA
#   MODEL_SIZE=$(get_model_size "$MODEL_NAME")

#   int_size=${MODEL_SIZE%%.*}
#   if (( int_size > 3 )); then
#     echo "Using LoRA for model size: $MODEL_SIZE"
#     PEFT=true
#   else
#     echo "Not using LoRA for model size: $MODEL_SIZE"
#     PEFT=false
#   fi
  
#   if ${PEFT}; then
#     PEFT_FLAGS="--peft --peft_path $MODEL_NAME"
#     MODEL_NAME="$BASE_MODEL_NAME"
#   else
#     PEFT_FLAGS=""
#   fi

  # Handle DS_SIZE parameter
  DS_SIZE_FLAG=""
  if [[ $DS_SIZE -gt -1 ]]; then
    echo "Limiting dataset size to: $DS_SIZE samples"
    DS_SIZE_FLAG="--ds_size $DS_SIZE"
  fi

  # Handle overwrite parameter
  if [ "$OVERWRITE" = true ]; then
    echo "Overwriting existing results"
    OVERWRITE_FLAG="--overwrite"
  else
    OVERWRITE_FLAG=""
  fi
  
  # Build command
  CMD="python src/eval_qa.py --model_name \"$MODEL_NAME\" --input_file \"$INPUT_FILE\" \
      --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
      --gpu_memory_utilization $GPU_MEMORY_UTIL \
      --max_new_tokens $MAX_NEW_TOKENS \
      --model_max_len $MODEL_MAX_LEN \
      $OVERWRITE_FLAG $PEFT_FLAGS $DS_SIZE_FLAG"
  
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

  # Add prediction mask
  if [ "$PREDICT_MASK" = true ]; then
    CMD="$CMD --predict_mask"
  fi
  
  # Print and execute command
  echo "Command: $CMD"
  eval "$CMD"
  echo "Finished at: $(date)"
  echo "---------------------------------------------"
done

echo "All evaluations completed at $(date)"
