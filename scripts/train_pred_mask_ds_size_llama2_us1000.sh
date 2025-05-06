#!/bin/bash

# Set the GPU device from the argument
export CUDA_VISIBLE_DEVICES=2,3

export WANDB_PROJECT="wikidyk-ar"

# Configuration variables (modify these according to your needs)
DATA_PATH="/data/yuwei/WikiDYK/data/wikidyk2022-2025_01082025_gpt-4o_evalv2_pages_formatted_combined_v2.json"
OUTPUT_DIR="train_results_pred_mask"
BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=2e-6
NUM_EPOCHS=1
MODEL_MAX_LENGTH=32768
CHAT_MODE=false  # Set to true for chat mode
NUM_UPSAMPLE=1000  # Default value for t5 models
QA_FORMAT_DATA_PATH=
QA_DATA_RATIO=-1  # Ratio of QA data to use
PREDICT_MASK=true

DS_SIZE_VALUES=(3500)

# infer nprocess_per_node from CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

# LoRA configuration
LORA_R=32
LORA_ALPHA=16

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define models to run
# You can add or remove models from this array
MODEL_NAMES=(
    # "downloaded_models/roberta-large"
    # "downloaded_models/t5-base"
    # "downloaded_models/flan-t5-xl"
    # "/data/yuwei/WikiDYK/downloaded_models/Llama-2-7b-hf"
    # "/data/yuwei/WikiDYK/downloaded_models/Qwen2.5-7B"
    # "downloaded_models/Qwen2.5-1.5B"
    "/data/yuwei/WikiDYK/downloaded_models/Llama-3.1-8B"
    # "downloaded_models/Qwen2.5-3B"
    # "downloaded_models/Qwen2.5-7B"
    # "meta-llama/Llama-2-7b-hf"
    # "meta-llama/Llama-3.2-3B"
    # "meta-llama/Llama-3.1-8B"
    # "Qwen/Qwen2.5-0.5B"
    # "Qwen/Qwen2.5-1.5B"
    # "Qwen/Qwen2.5-3B"
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-14B"
)

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

# Loop through each model
for MODEL_NAME in "${MODEL_NAMES[@]}"; do

  for DS_SIZE in "${DS_SIZE_VALUES[@]}"; do
  
    # Create model-specific output directory
    MODEL_OUTPUT_DIR="$OUTPUT_DIR/${MODEL_NAME//\//_}"

    # Add ds_size to directory name if specified
    # if -1 then add full dataset size
    if [[ $DS_SIZE -gt 0 ]]; then
      MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}_ds${DS_SIZE}"
    elif [[ $DS_SIZE -eq -1 ]]; then
      MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}_full"
    else
      MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}_ds0"
    fi

    # Add upsample size to directory name if specified
    if [[ $NUM_UPSAMPLE -gt 0 ]]; then
      MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}_upsample${NUM_UPSAMPLE}"
    fi

    # Add QA data ratio to directory name if specified
    if [[ $QA_DATA_RATIO -gt 0 ]]; then
      MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}_qa${QA_DATA_RATIO}"
    fi

    # Add predict mask to directory name if specified
    if [[ $PREDICT_MASK == true ]]; then
      MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}_predict_mask"
    fi
  
    # Create model-specific logs
    MODEL_LOG="${MODEL_OUTPUT_DIR}/${TIMESTAMP}.log"
    MODEL_ERROR_LOG="${MODEL_OUTPUT_DIR}/${TIMESTAMP}.log"

    # Function for logging
    log() {
      local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
      echo "$message" | tee -a "$MODEL_LOG"
    }

    error_log() {
      local message="[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1"
      echo "$message" | tee -a "$MODEL_LOG" "$MODEL_ERROR_LOG"
    }
    
    mkdir -p "$MODEL_OUTPUT_DIR"
    log "Created output directory: $MODEL_OUTPUT_DIR"
  
    # Set chat mode flag based on model type and CHAT_MODE variable
    CHAT_FLAG=""
    if [[ "$CHAT_MODE" == true ]]; then
      CHAT_FLAG="--chat_mode"
      log "Chat mode enabled"
    else
      log "Chat mode disabled"
    fi

    if [[ "$MODEL_NAME" == *"Llama-2"* ]]; then
      MODEL_MAX_LENGTH=4096
      log "Set MODEL_MAX_LENGTH to 4096 for Llama-2 model"
    fi
    
    # Check if model size is over 3B to use LoRA
    MODEL_SIZE=$(get_model_size "$MODEL_NAME")
    LORA_FLAGS=""
  
    if (( $(echo "$MODEL_SIZE > 3" | bc -l) )); then
      log "Model size is over 3B ($MODEL_SIZE B). Using LoRA training."
      LORA_FLAGS="--use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA"
      LEARNING_RATE=2e-4
      log "Adjusted learning rate for LoRA: $LEARNING_RATE"
    else
      log "Model size is 3B or smaller ($MODEL_SIZE B). Using full fine-tuning."
    fi

    if [[ "$MODEL_NAME" == *"t5-base"* ]]; then
      LEARNING_RATE=3e-4
      BATCH_SIZE=512
      GRADIENT_ACCUMULATION_STEPS=1
      log "Adjusted parameters for t5 model:"
      log "  - LEARNING_RATE: $LEARNING_RATE"
      log "  - BATCH_SIZE: $BATCH_SIZE"
      log "  - GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    elif [[ "$MODEL_NAME" == *"t5-large"* ]]; then
      LEARNING_RATE=1e-4
      BATCH_SIZE=128
      GRADIENT_ACCUMULATION_STEPS=1
      log "Adjusted parameters for t5 model:"
      log "  - LEARNING_RATE: $LEARNING_RATE"
      log "  - BATCH_SIZE: $BATCH_SIZE"
      log "  - GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    elif [[ "$MODEL_NAME" == *"t5-xl"* ]]; then
      LEARNING_RATE=3e-4
      BATCH_SIZE=32
      GRADIENT_ACCUMULATION_STEPS=1
      log "Adjusted parameters for t5 model:"
      log "  - LEARNING_RATE: $LEARNING_RATE"
      log "  - BATCH_SIZE: $BATCH_SIZE"
      log "  - GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    elif [[ "$MODEL_NAME" == *"t5-xxl"* ]]; then
      LEARNING_RATE=3e-4
      BATCH_SIZE=32
      GRADIENT_ACCUMULATION_STEPS=1
      log "Adjusted parameters for t5 model:"
      log "  - LEARNING_RATE: $LEARNING_RATE"
      log "  - BATCH_SIZE: $BATCH_SIZE"
      log "  - GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    elif [[ "$MODEL_NAME" == *"t5-small"* ]]; then
      LEARNING_RATE=1e-4
      BATCH_SIZE=512
      GRADIENT_ACCUMULATION_STEPS=1
      log "Adjusted parameters for t5 model:"
      log "  - LEARNING_RATE: $LEARNING_RATE"
      log "  - BATCH_SIZE: $BATCH_SIZE"
      log "  - GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    fi

    if [[ "$MODEL_NAME" == *"roberta-base"* ]]; then
      LEARNING_RATE=1e-5
      BATCH_SIZE=1024
      GRADIENT_ACCUMULATION_STEPS=1
      log "Adjusted parameters for roberta model:"
      log "  - LEARNING_RATE: $LEARNING_RATE"
      log "  - BATCH_SIZE: $BATCH_SIZE"
      log "  - GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    elif [[ "$MODEL_NAME" == *"roberta-large"* ]]; then
      LEARNING_RATE=1e-5
      BATCH_SIZE=512
      GRADIENT_ACCUMULATION_STEPS=1
      log "Adjusted parameters for roberta model:"
      log "  - LEARNING_RATE: $LEARNING_RATE"
      log "  - BATCH_SIZE: $BATCH_SIZE"
      log "  - GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    fi

    if [[ "$MODEL_NAME" == *"Llama-3.2-1B"* ]]; then
      BATCH_SIZE=128
      GRADIENT_ACCUMULATION_STEPS=1
      log "Adjusted parameters for Llama model:"
      log "  - LEARNING_RATE: $LEARNING_RATE"
      log "  - BATCH_SIZE: $BATCH_SIZE"
      log "  - GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    fi
  
    # Handle QA_FORMAT_DATA_PATH
    QA_FORMAT_FLAG=""
    if [[ "$QA_FORMAT_DATA_PATH" != "None" && -n "$QA_FORMAT_DATA_PATH" ]]; then
      log "Using QA format data from: $QA_FORMAT_DATA_PATH"
      log "Using QA data ratio: $QA_DATA_RATIO"
      QA_FORMAT_FLAG="--qa_format_data_path $QA_FORMAT_DATA_PATH"
    else
      log "No QA format data will be used"
    fi
  
    # Handle DS_SIZE parameter
    DS_SIZE_FLAG=""
    if [[ $DS_SIZE -gt -1 ]]; then
      log "Limiting dataset size to: $DS_SIZE samples"
      DS_SIZE_FLAG="--ds_size $DS_SIZE"
    fi

    # Start model training section in log
    log "======================================="
    log "Starting training for model: $MODEL_NAME"
    log "======================================="
    # calculate global batch size
    GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
    log "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    log "WANDB_PROJECT: $WANDB_PROJECT"
    log "DATA_PATH: $DATA_PATH"
    log "Global Batch Size: $GLOBAL_BATCH_SIZE"
    log "Data Size: $DS_SIZE"
  
    # Log the full command
    TRAIN_CMD="torchrun --nproc_per_node \"$NUM_GPUS\" --master-port 29581 src/train.py \
      --model_name_or_path \"$MODEL_NAME\" \
      --data_path \"$DATA_PATH\" \
      --output_dir \"$MODEL_OUTPUT_DIR\" \
      --num_upsample \"$NUM_UPSAMPLE\" \
      --per_device_train_batch_size \"$BATCH_SIZE\" \
      --gradient_accumulation_steps \"$GRADIENT_ACCUMULATION_STEPS\" \
      --learning_rate \"$LEARNING_RATE\" \
      --num_train_epochs \"$NUM_EPOCHS\" \
      --model_max_length \"$MODEL_MAX_LENGTH\" \
      --report_to wandb --logging_steps 50 \
      --save_strategy steps --save_steps 10000 \
      --save_total_limit 3 \
      --resume_from_checkpoint True \
      --bf16 True --use_flash_attention_2 True \
      --qa_data_ratio \"$QA_DATA_RATIO\" \
      --predict_mask \"$PREDICT_MASK\" \
      ${CHAT_FLAG:+$CHAT_FLAG} \
      ${LORA_FLAGS:+$LORA_FLAGS} \
      ${QA_FORMAT_FLAG:+$QA_FORMAT_FLAG} \
      ${DS_SIZE_FLAG:+$DS_SIZE_FLAG}"
    log "Executing command: $TRAIN_CMD"
  
    # Run the Python script with appropriate parameters and capture output
    log "Training started at $(date)"
  
    # Execute and log both stdout and stderr
    {
      eval $TRAIN_CMD
    } > >(tee -a "$MODEL_LOG") 2> >(tee -a "$MODEL_ERROR_LOG" >&2)
  
    TRAIN_STATUS=$?
  
    # Check if the command was successful
    if [ $TRAIN_STATUS -eq 0 ]; then
      log "Training completed successfully for $MODEL_NAME"
    else
      error_log "Training failed for $MODEL_NAME with exit code $TRAIN_STATUS"
      log "Check error log for details: $MODEL_ERROR_LOG"
    fi
  
    # Log resource utilization
    log "Resource usage after training $MODEL_NAME:"
    log "GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$MODEL_LOG"
  
    log "Disk space usage for model outputs:"
    du -sh "$MODEL_OUTPUT_DIR" | tee -a "$MODEL_LOG"
  
    log ""
  done

  log "All training runs completed at $(date)"
  log "======================================="
  log "Summary of training runs:"

  # Generate training summary
  log "Model | Status | Duration | Output Size"
  for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_OUTPUT_DIR="$OUTPUT_DIR/${MODEL_NAME//\//_}"
    if [[ $DS_SIZE -gt 0 ]]; then
      MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR}_ds${DS_SIZE}"
    fi
  
    MODEL_LOG="$MODEL_OUTPUT_DIR/${TIMESTAMP}.log"
    if [ -f "$MODEL_LOG" ]; then
      START_TIME=$(head -n 20 "$MODEL_LOG" | grep "Training started" | tail -n 1 | awk -F'at ' '{print $2}')
      END_TIME=$(tail -n 20 "$MODEL_LOG" | grep "Training completed\|Training failed" | head -n 1 | awk -F'at ' '{print $2}')
    
      if grep -q "Training completed successfully" "$MODEL_LOG"; then
        STATUS="SUCCESS"
      else
        STATUS="FAILED"
      fi
    
      OUTPUT_SIZE=$(du -sh "$MODEL_OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
    
      log "$MODEL_NAME | $STATUS | $START_TIME to $END_TIME | $OUTPUT_SIZE"
    else
      log "$MODEL_NAME | UNKNOWN | N/A | N/A"
    fi
  done
done

log "Log files are available in: $MODEL_OUTPUT_DIR"
log "======================================="