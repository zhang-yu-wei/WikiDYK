import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from transformers import Trainer, TrainingArguments, PreTrainedTokenizer, HfArgumentParser, AutoTokenizer
from peft import LoraConfig, get_peft_model

from utils.tools import set_all_seeds, get_model_type, load_model
from utils.dataloading import (
    SupervisedDataset,
    DataCollatorForSupervisedDataset
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=16)
    use_flash_attention_2: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    years: str = field(default="all", metadata={"help": "Years to filter the data."})
    chat_mode: bool = field(default=False, metadata={"help": "Whether to use chat mode."})
    num_upsample: int = field(default=1, metadata={"help": "Number of times to upsample the data."})
    use_page: bool = field(default=False, metadata={"help": "Whether to use page data."})
    qa_format_data_path: str = field(default=None, metadata={"help": "Path to the QA format data."})
    ds_size: int = field(default=None, metadata={"help": "Size of the dataset to use."})
    qa_data_ratio: float = field(default=0.5, metadata={"help": "Ratio of QA data to use."})
    predict_mask: bool = field(default=False, metadata={"help": "Whether to predict mask."})

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args, model_type: str) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=data_args.data_path, 
        years=data_args.years,
        chat_mode=data_args.chat_mode, 
        num_upsample=data_args.num_upsample,
        model_type=model_type,
        use_page=data_args.use_page,
        qa_format_data_path=data_args.qa_format_data_path,
        ds_size=data_args.ds_size,
        qa_data_ratio=data_args.qa_data_ratio,
        predict_mask=data_args.predict_mask,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, model_type=model_type)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seeds for reproducibility
    set_all_seeds(training_args.seed)
    
    # Determine model type for specialized processing
    model_type = get_model_type(model_args.model_name_or_path)
    if model_type != "gpt":
        data_args.predict_mask = False
    
    logging.warning(f"Output directory: {training_args.output_dir}")

    # ==== Model and Tokenizer Loading ====
    model = load_model(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_flash_attention_2=model_args.use_flash_attention_2,
    )

    if model_args.use_lora:
        task_type = "CAUSAL_LM" if model_type == "gpt" else "SEQ_2_SEQ_LM" if model_type == 't5' else "MASKED_LM"
        peft_config = LoraConfig(
            task_type=task_type, 
            inference_mode=False, 
            r=model_args.lora_r, 
            lora_alpha=model_args.lora_alpha
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left" if "qwen" in model_args.model_name_or_path.lower() else "right",
        use_fast=False,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    # ==== data module ====
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        data_args=data_args,
        model_type=model_type
    )
    
    # ==== Training ====
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)
    trainer.push_to_hub()


if __name__ == "__main__":
    train()