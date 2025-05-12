#!/bin/bash

# Language Model Evaluation Script for Multiple Models
# This script runs the language model evaluation on WikiDYK data for multiple models
export CUDA_VISIBLE_DEVICES="2,3"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Input file and common parameters
# INPUT_FILE="converted_questions/Qwen-Qwen2.5-7B-Instruct_evalds100/wikidyk2022-2025_01082025_gpt-4o_evalv2_pages_formatted_combined_v2_eval.json"
INPUT_FILE="data/wikidyk2022-2025_01082025_gpt-4o_evalv2_pages_formatted_combined_v2.json"
OUTPUT_DIR="eval_results"
# Infer tensor parallel size and visible devices
TENSOR_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE + 1))
GPU_MEMORY_UTIL=0.80
MAX_NEW_TOKENS=256
MODEL_MAX_LEN=1024
RAG_TOP_K=0
USE_CHAT_MODE=false
DS_SIZE=-1
PREDICT_MASK=false
OVERWRITE=false
PEFT=false

# Models to evaluate
MODELS=(
    "YWZBrandon/google_flan-t5-base_semantic_10_clusters_9_full_upsample1000"
    "YWZBrandon/google_flan-t5-base_temporal_5_clusters_3_full_upsample1000"
    "YWZBrandon/google_flan-t5-base_temporal_5_clusters_4_full_upsample1000"
    "YWZBrandon/google_flan-t5-base_temporal_10_clusters_0_full_upsample1000"
    "YWZBrandon/google_flan-t5-base_temporal_10_clusters_1_full_upsample1000"
)
BASE_MODEL_NAME=""

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

  # Check if model size is over 3B to use LoRA
  MODEL_SIZE=$(get_model_size "$MODEL_NAME")
  
  if ${PEFT}; then
    PEFT_FLAGS="--peft --peft_path $MODEL_NAME"
    MODEL_NAME="$BASE_MODEL_NAME"
  else
    PEFT_FLAGS=""
  fi

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
  CMD="accelerate launch \
      --num_processes $TENSOR_PARALLEL_SIZE \
      --num_machines 1 \
      --main_process_port 29500 \
      src/eval_qa_accelerate.py --model_name \"$MODEL_NAME\" --input_file \"$INPUT_FILE\" \
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