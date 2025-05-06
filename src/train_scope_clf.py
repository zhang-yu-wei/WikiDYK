import numpy as np
import random
import torch
import json
import argparse
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from transformers import DataCollatorWithPadding
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    TrainingArguments, Trainer, HfArgumentParser
)
from datasets import load_from_disk
import evaluate
import warnings
warnings.filterwarnings('ignore')

from utils.tools import set_all_seeds

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="microsoft/deberta-v3-large")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    evaluate_only: bool = field(default=False, metadata={"help": "Whether to evaluate only."})

def preprocess_function(example, class2id, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(
        example['text'], truncation=True
    )
    
    # Convert labels to IDs
    labels = [0.] * len(class2id) # float because bce loss only supports float
    if example['cluster_id'] >= 0:
        labels[class2id[example['cluster_id']]] = 1
    
    inputs['labels'] = labels
    return inputs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))

def check_model_use_flash_attention_2(model_name):
    for name in ['llama', 'qwen', 'gemma']:
        if name in model_name.lower():
            return True

# Main function to run the training and evaluation
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_all_seeds(training_args.seed)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load your dataset
    dataset = load_from_disk(data_args.data_path)
    all_labels = list(set(dataset['train']['cluster_id']))
    all_labels.sort()
    id2class = {i: i for i in all_labels if i != -1}
    class2id = {i: i for i in all_labels if i != -1}
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, class2id, tokenizer),
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    # Initialize model
    use_flash_attention_2 = check_model_use_flash_attention_2(model_args.model_name_or_path)
    num_labels = len(all_labels)-1 # Exclude the -1 label
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=num_labels,
        id2label=id2class, label2id=class2id,
        problem_type = "multi_label_classification",
        attn_implementation="flash_attention_2" if use_flash_attention_2 else "eager",
        torch_dtype=torch.bfloat16 if use_flash_attention_2 else torch.float32,
    )
    model.to('cuda')

    if 'llama' in model_args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    
    # Training loop
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['eval'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if data_args.evaluate_only:
        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)
        return
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()