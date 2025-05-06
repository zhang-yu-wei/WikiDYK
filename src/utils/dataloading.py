import copy
import json
import logging
import random
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, Sequence
from transformers import PreTrainedTokenizer
from utils.mask_strategy import sample_masked_versions, create_mlm_input, create_qa_mlm_input

IGNORE_INDEX = -100
PROMPT_TEMPLATE = "{input_str}\nAnswer:"
AR_MASK_PREDICT_PROMPT = "Predict the masked words in the following sentence: {input_str}\nMasked words:\n"

def is_mask_prediction_task(model_type: str, predict_mask: bool) -> bool:
    """
    Determine if the task is a masked prediction task based on the model type and predict_mask flag.
    
    Args:
        model_type (str): The type of the model (e.g., "t5", "bert", "gpt").
        predict_mask (bool): Flag indicating if masked prediction is used.
        
    Returns:
        bool: True if the task is a masked prediction task, False otherwise.
    """
    return (model_type.lower() in ["t5"]) or (model_type.lower() == "gpt" and predict_mask)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with proper shuffling."""

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, 
             years: str = "all", chat_mode: bool = False, num_upsample: int = 1,
             model_type: str = "", use_page: bool = False, qa_format_data_path: str = None,
             qa_data_ratio: float = 0.5, ds_size: int = None, predict_mask: bool = False):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, 'r') as f:
            raw_data_list = json.load(f)
            if ds_size is not None:
                raw_data_list = raw_data_list[:ds_size]
        
        # Process and store the raw data
        self.model_lower = model_type.lower()
        self.tokenizer = tokenizer
        self.chat_mode = chat_mode
        self.num_upsample = num_upsample
        self.use_page = use_page
        self.predict_mask = predict_mask
        
        # Store raw data for on-the-fly processing
        self.raw_texts = []

        # Process QA data if provided
        self.qa_questions = []
        self.qa_answers = []
        has_qa_data = False

        # Add training QA data which is different from the additional QA data
        training_qa_questions = []
        training_qa_answers = []

        years = years.split(",") if years != "all" else years

        for example in raw_data_list:
            
            if example.get("year") not in years and years != "all":
                continue

            if use_page:
                fact_text = example['bold_entity_page']['content']
            else:
                fact_text = example['fact']
                if fact_text.startswith("that"):
                    fact_text = fact_text[4:]
                if fact_text.endswith("?"):
                    fact_text = fact_text[:-1]
            self.raw_texts.append(fact_text.strip())

            # NEW: Include paraphrases if they exist
            if "paraphrases" in example and isinstance(example["paraphrases"], list):
                for paraphrase in example["paraphrases"]:
                    if "paraphrased_fact" in paraphrase:
                        self.raw_texts.append(paraphrase["paraphrased_fact"])
            
            # NEW: Include reordered facts if they exist
            if "reordered_facts" in example and isinstance(example["reordered_facts"], list):
                for reordered in example["reordered_facts"]:
                    if "reordered_fact" in reordered:
                        self.raw_texts.append(reordered["reordered_fact"])
            
            if "generated_qas" in example and isinstance(example["generated_qas"], list):
                training_qas = []
                for qa in example["generated_qas"]:
                    if "question" in qa and "answer" in qa:
                        question = qa["question"]
                        answer = qa["answer"]
                        training_qas.append((question, answer))
                if self.num_upsample > 1:
                    while len(training_qas) < self.num_upsample:
                        training_qas.extend(training_qas[:min(len(training_qas), self.num_upsample - len(training_qas))])
                training_qa_questions.extend([qa[0] for qa in training_qas])
                training_qa_answers.extend([qa[1] for qa in training_qas])
        
        format_qa_questions = []
        format_qa_answers = []
        if qa_format_data_path is not None:
            logging.warning("Loading QA format data...")
            with open(qa_format_data_path, 'r') as f:
                format_qa_data_list = json.load(f)
            
            for example in format_qa_data_list:
                question = example['question']
                answer = example['answer']
                format_qa_questions.append(question)
                format_qa_answers.append(answer)
        
        # Integrate QA training data
        self.qa_questions = training_qa_questions + format_qa_questions
        self.qa_answers = training_qa_answers + format_qa_answers

        has_qa_data = len(self.qa_questions) > 0
        
        # Adjust the dataset size for upsampling and QA data
        self.base_size = len(self.raw_texts)

        # Create combined indices for proper shuffling
        self.combined_indices = []

        mask_prediction_task = is_mask_prediction_task(self.model_lower, self.predict_mask)
        
        # Always add fact indices with upsampling
        self.all_upsampled_texts = []
        for i in range(self.base_size):
            if self.num_upsample > 1 and mask_prediction_task:
                mask_token = "<extra_id_0>" if self.model_lower == "t5" else "[MASK]"
                upsampled_texts = sample_masked_versions(self.raw_texts[i], self.num_upsample, mask_token=mask_token)
                self.all_upsampled_texts.extend(upsampled_texts)
                for j in range(self.num_upsample):
                    self.combined_indices.append(("fact", i, j))
            else:
                # append all copy of the text
                self.all_upsampled_texts.extend([self.raw_texts[i]] * self.num_upsample)
                for j in range(self.num_upsample):
                    # add the index of the fact text
                    self.combined_indices.append(("fact", i, j))
                
        # If QA data is present, calculate effective dataset size
        if has_qa_data and qa_data_ratio > 0:
            qa_size = len(self.qa_questions)
            # Calculate the upsampled fact data size
            upsampled_fact_size = self.base_size * self.num_upsample
            
            # Calculate target sizes based on the requested ratio
            # If we want x% of data to be QA, and (1-x)% to be fact:
            # fact_count / (fact_count + qa_count) = (1-x)
            # Solving for qa_count:
            # qa_count = fact_count * x / (1-x)
            target_qa_count = int(upsampled_fact_size * qa_data_ratio / (1 - qa_data_ratio))
            
            # Calculate how many times we need to repeat each QA example
            # to achieve the target ratio (but not exceed the available QA examples)
            qa_repeat_factor = max(1, int(target_qa_count / qa_size))
            qa_remainder = target_qa_count - (qa_size * qa_repeat_factor)
                
            # Add QA indices with proper sampling to reach target ratio
            for repeat in range(qa_repeat_factor):
                for i in range(qa_size):
                    self.combined_indices.append(("qa", i))
            
            # Add remaining QA indices to reach exact target ratio
            if qa_remainder > 0:
                additional_qa_indices = random.sample(range(qa_size), qa_remainder)
                for i in additional_qa_indices:
                    self.combined_indices.append(("qa", i))
            
            logging.warning(f"Dataset initialized with QA data:")
            logging.warning(f"  - {upsampled_fact_size} fact examples (with upsampling factor {self.num_upsample})")
            logging.warning(f"  - {qa_size * qa_repeat_factor + qa_remainder} QA examples")
            logging.warning(f"  - QA ratio target: {qa_data_ratio:.3f}")
            logging.warning(f"  - Actual QA ratio: {(qa_size * qa_repeat_factor + qa_remainder) / (upsampled_fact_size + qa_size * qa_repeat_factor + qa_remainder):.3f}")
                        
        elif qa_data_ratio == -1:
            # integrate all qa data
            for i in range(len(self.qa_questions)):
                self.combined_indices.append(("qa", i))
            logging.warning(f"Dataset initialized with all QA data:")
            logging.warning(f"  - {len(self.qa_questions)} QA examples")
            logging.warning(f"  - {self.base_size} fact examples with upsampling factor {self.num_upsample}")
            logging.warning(f"  - Total examples: {len(self.combined_indices)}")
        else:
            # No QA data or ratio is 0, only fact data
            logging.warning(f"Dataset initialized with fact data only:")
            logging.warning(f"  - {self.base_size} fact examples with upsampling factor {self.num_upsample}")
            logging.warning(f"  - Total examples: {self.base_size * self.num_upsample}")
            
        # Shuffle the combined indices
        random.shuffle(self.combined_indices)
        self.size = len(self.combined_indices)
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Use our properly shuffled combined indices
        index_info = self.combined_indices[idx]
        
        if index_info[0] == "fact":
            _, fact_idx, upsample_idx = index_info
            return self._process_fact_example(fact_idx, upsample_idx)
        else:  # index_info[0] == "qa"
            _, qa_idx = index_info
            return self._process_qa_example(qa_idx)
    
    def _process_fact_example(self, fact_idx, upsample_idx):
        fact_text = self.raw_texts[fact_idx]
    
        if self.model_lower == "t5":
            overall_idx = fact_idx * self.num_upsample + upsample_idx
            corrupted_text, target_text = self.all_upsampled_texts[overall_idx]
            # Format the target text for T5
            target_text = "<extra_id_0> " + target_text + " <extra_id_1>"
            return self._tokenize_t5_example(corrupted_text, target_text)
        
        elif self.model_lower == "bert":            
            fact_tokenized = self.tokenizer(
                fact_text,
                return_tensors="pt",
                padding=False,  # No padding here
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                add_special_tokens=True,
            )
            input_ids = fact_tokenized.input_ids[0]
            # Create MLM input with different random seed for diversity
            input_id, label = create_mlm_input(input_ids, self.tokenizer, seed=upsample_idx, mlm_probability=0.3)
            return {"input_ids": input_id, "labels": label}
        
        elif self.model_lower == "gpt":
            # For decoder-only models
            if self.predict_mask:
                # For GPT models, generate a masked prediction prompt
                overall_idx = fact_idx * self.num_upsample + upsample_idx
                corrupted_text, target_text = self.all_upsampled_texts[overall_idx]
                # Create the prompt for masked prediction
                prompt_text = AR_MASK_PREDICT_PROMPT.format(input_str=corrupted_text)
                return self._tokenize_ar_example(prompt_text, target_text)
            else:
                overall_idx = fact_idx * self.num_upsample + upsample_idx
                fact_text = self.all_upsampled_texts[overall_idx]
                return self._tokenize_ar_example(fact_text)
        else:
            raise ValueError(f"Unsupported model type: {self.model_lower}. Supported types are: t5, bert, gpt.")
    
    def _process_qa_example(self, qa_idx):
        # TODO: modify this code to handle predict mask
        question = self.qa_questions[qa_idx]
        # Sample one answer if multiple are provided
        answer_options = self.qa_answers[qa_idx]
        answer = random.choice(answer_options) if isinstance(answer_options, list) else answer_options
                
        if self.model_lower == "t5":
            # For T5/BART: input is the question with a mask token, target is the answer
            input_text = PROMPT_TEMPLATE.format(input_str=question)
            input_text += " <extra_id_0>"  # Add mask token for T5
            target_text = "<extra_id_0> " + answer + " <extra_id_1>"  # Format expected by T5
            return self._tokenize_t5_example(input_text, target_text)

        elif self.model_lower == "bert":
            # For BERT: input is the question with a mask token, target is the answer
            input_text = PROMPT_TEMPLATE.format(input_str=question)
            input_tokenized = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=False,  # No padding here
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                add_special_tokens=True,
            )['input_ids'][0][:-1]
            target_tokenized = self.tokenizer(
                answer,
                return_tensors="pt",
                padding=False,  # No padding here
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                add_special_tokens=False,
            )['input_ids'][0]
            target_tokenized = torch.cat([target_tokenized, torch.tensor([self.tokenizer.eos_token_id])])
            input_ids, labels = create_qa_mlm_input(input_tokenized, target_tokenized, self.tokenizer)
            return {"input_ids": input_ids, "labels": labels}
            
        elif self.model_lower == "gpt":
            input_text = PROMPT_TEMPLATE.format(input_str=question)
            if self.predict_mask:
                input_text = AR_MASK_PREDICT_PROMPT.format(input_str=input_text + " [MASK]")
                return self._tokenize_ar_example(input_text, answer)
            else:
                return self._tokenize_ar_example(input_text, answer)
        else:
            raise ValueError(f"Unsupported model type: {self.model_lower}. Supported types are: t5, bert, gpt.")
    
    def _tokenize_t5_example(self, input_text, target_text):
        # Tokenize input without padding to max_length
        input_tokens = self.tokenizer(
            input_text,
            padding=False,  # No padding here
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        
        # Tokenize target without padding to max_length
        if hasattr(self.tokenizer, 'as_target_tokenizer'):
            with self.tokenizer.as_target_tokenizer():
                target_tokens = self.tokenizer(
                    target_text,
                    padding=False,  # No padding here
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
        else:
            target_tokens = self.tokenizer(
                target_text,
                padding=False,  # No padding here
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
        
        # Replace padding token id with IGNORE_INDEX in labels
        labels = target_tokens.input_ids[0].clone()
        
        return {
            "input_ids": input_tokens.input_ids[0],
            "attention_mask": input_tokens.attention_mask[0],
            "labels": labels
        }

    def _tokenize_ar_example(self, prompt_text, target_text=None):
        if target_text is None:
            assert not self.chat_mode, "target_text should not be None in chat mode"
            if self.tokenizer.bos_token is not None:
                text = self.tokenizer.bos_token + prompt_text
            else:
                text = prompt_text
            
            tokenized = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,  # No padding here
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                add_special_tokens=False,
            )
            input_ids = tokenized.input_ids[0]
            
            # For decoder-only, labels are the same as input_ids
            return {"input_ids": input_ids, "labels": input_ids.clone()}
        else:
            # ensure that loss is only calculated for the target text
            if self.chat_mode:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    all_text = self.tokenizer.apply_chat_template([
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": target_text}
                    ], tokenize=False, add_generation_prompt=True)
                    prompt_text = self.tokenizer.apply_chat_template([
                        {"role": "user", "content": prompt_text},
                    ], tokenize=False, add_generation_prompt=True)
                else:
                    # Fallback for tokenizers without chat template
                    all_text = f"<s>[INST] {prompt_text} [/INST] {target_text}</s>"
                    prompt_text = f"<s>[INST] {prompt_text} [/INST]"
            else:
                if self.tokenizer.bos_token is not None:
                    all_text = self.tokenizer.bos_token + prompt_text + target_text
                    prompt_text = self.tokenizer.bos_token + prompt_text
                else:
                    all_text = prompt_text + target_text
                
                all_text += self.tokenizer.eos_token if 'qwen' not in self.tokenizer.name_or_path.lower() else ""

            # Tokenize without padding to max_length
            tokenized = self.tokenizer(
                all_text,
                return_tensors="pt",
                padding=False,  # No padding here
                truncation=True,
                add_special_tokens=False,
                max_length=self.tokenizer.model_max_length,
            )
            input_ids = tokenized.input_ids[0]
            labels = input_ids.clone()
            # Set all tokens before the target text to -100 (IGNORE_INDEX)
            # find the end of the prompt
            tokenized_prompt = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=False,  # No padding here
                truncation=True,
                add_special_tokens=False,
                max_length=self.tokenizer.model_max_length,
            )
            prompt_end = tokenized_prompt.input_ids[0].size(0)
            # Set all tokens before the target text to -100 (IGNORE_INDEX)
            labels[:prompt_end] = IGNORE_INDEX
            return {"input_ids": input_ids, "labels": labels}

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning with dynamic padding."""

    tokenizer: PreTrainedTokenizer
    model_type: str = ""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        model_lower = self.model_type.lower()
        
        # Get all available keys
        keys = instances[0].keys()
        batch = {}
        
        for key in keys:
            if key == "labels" and isinstance(instances[0][key], list):
                # Handle labels specially for T5/BART
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(instance[key]) for instance in instances],
                    batch_first=True,
                    padding_value=IGNORE_INDEX
                )
            else:
                # Dynamically pad tensors to the maximum length in the current batch
                values = [instance[key] for instance in instances]
                if isinstance(values[0], torch.Tensor):
                    padding_value = self.tokenizer.pad_token_id if key == "input_ids" else 0
                    batch[key] = torch.nn.utils.rnn.pad_sequence(
                        values, batch_first=True, padding_value=padding_value,
                        padding_side="left" if model_lower == "gpt" else "right"
                    )
        
        # Add attention mask if not present
        if "attention_mask" not in batch:
            batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id)
        
        # Debug info - helpful during development
        # print(f"Batch input_ids shape: {batch['input_ids'].shape}")
            
        return batch