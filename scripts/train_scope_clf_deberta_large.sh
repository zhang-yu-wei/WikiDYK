export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="wikidyk-scope-clf"

MODEL_NAME=""YWZBrandon/wikidyk-scope-clf-deberta-v3-large-semantic_3_clusters""

# infer nprocess_per_node from CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

EVAL_ONLY=true
EVAL_ONLY_FLAG=""
if [ "$EVAL_ONLY" = true ]; then
    EVAL_ONLY_FLAG="--evaluate_only"
fi

data_types=(
    "semantic_3_clusters"
)

for data_type in "${data_types[@]}"; do
    DATA_PATH="data/scope_clf_data/$data_type"
    WANDB_RUN_NAME="wikidyk-scope-clf-deberta-v3-large-$data_type"
    if [ "$EVAL_ONLY" = true ]; then
        WANDB_RUN_NAME="wikidyk-scope-clf-deberta-v3-large-$data_type-eval-only"
    fi
    OUTPUT_DIR="train_results/$WANDB_RUN_NAME"
    LEARNING_RATE=2e-5
    BATCH_SIZE=32
    NUM_EPOCHs=10

    torchrun --nproc_per_node=$NUM_GPUS --master_port 12345 \
        src/train_scope_clf.py \
        --model_name_or_path $MODEL_NAME \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --learning_rate $LEARNING_RATE \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --num_train_epochs $NUM_EPOCHs \
        --gradient_accumulation_steps 1 \
        --eval_strategy "epoch" \
        --save_strategy "epoch" \
        --logging_steps 10 \
        --make_prediction \
        ${EVAL_ONLY_FLAG:+$EVAL_ONLY_FLAG}
done