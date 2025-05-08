export CUDA_VISIBLE_DEVICES=3
export WANDB_PROJECT="wikidyk-scope-clf"

for data_type in "semantic_3_clusters"; do
# for data_type in "semantic_5_clusters" "temporal_5_clusters"; do
    # MODEL_NAME="microsoft/deberta-v3-large"
    MODEL_NAME="train_results/wikidyk-scope-clf-deberta-v3-large-semantic_3_clusters"
    DATA_PATH="data/scope_clf_data/$data_type"
    WANDB_RUN_NAME="wikidyk-scope-clf-deberta-v3-large-$data_type"
    OUTPUT_DIR="train_results/$WANDB_RUN_NAME"
    LEARNING_RATE=2e-5
    BATCH_SIZE=16
    NUM_EPOCHs=10

    python src/train_scope_clf.py \
        --model_name_or_path $MODEL_NAME \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --learning_rate $LEARNING_RATE \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --num_train_epochs $NUM_EPOCHs \
        --gradient_accumulation_steps 4 \
        --eval_strategy "epoch" \
        --save_strategy "epoch" \
        --logging_steps 10 \
        --evaluate_only \
        --make_prediction
done