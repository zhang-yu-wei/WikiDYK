#!/usr/bin/env python3
"""
Language Model Evaluation Script for Question Answering Tasks

This script evaluates language models on question answering tasks using WikiDYK data.
It supports various models including OpenAI models and locally hosted models via vLLM.
"""

import argparse
import json
import os
import glob
from datetime import datetime
from shutil import rmtree
from typing import Dict, List, Tuple, Union, Any, Optional

import torch
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
from vllm import LLM, SamplingParams
from accelerate import Accelerator
from accelerate.utils import gather_object
accelerator = Accelerator()

from utils.metrics import compare
from utils.tools import set_all_seeds, openai_async_inference
from utils.table_generator import table_generator

# Constants
OPENAI_API_KEY = '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'
DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
PROMPT_TEMPLATE = "{input_str}\nAnswer:"
PROMPT_TEMPLATE_RAG = "{context}\n\nBased on the above contexts, answer the following question:{input_str}\nAnswer:"
AR_MASK_PREDICT_PROMPT = "Predict the masked words in the following sentence: {input_str}\nMasked words:\n"

# Type definitions for clarity
class Example:
    def __init__(self, input_str: str, expected_output: str, question: str, fact: str, date: str, type: str, case_id: Optional[str] = None):
        self.input_str = input_str
        self.expected_output = expected_output
        self.question = question
        self.fact = fact
        self.date = date
        self.type = type
        self.case_id = case_id

Result = Dict[str, Any]


def retrieve_page(
    query_embedding: torch.Tensor, 
    embeddings: torch.Tensor, 
    datastore: List[str], 
    k: int = 1
) -> List[str]:
    """
    Retrieve relevant pages from datastore based on embedding similarity.
    
    Args:
        query_embedding: Embedding of the query
        embeddings: Embeddings of all documents in the datastore
        datastore: List of document texts
        k: Number of documents to retrieve
        
    Returns:
        List of retrieved document texts
    """
    # Calculate cosine similarity
    scores = query_embedding @ embeddings.T
    
    # Get top-k indices
    _, indices = torch.topk(scores, k)
    
    # Return corresponding documents
    return [datastore[i] for i in indices]


def merge_lora_weights(base_model_name: str, lora_path: str) -> None:
    """
    Merge LoRA weights into the base model.
    
    Args:
        base_model_name: Name/path of the base model
        lora_path: Path to the LoRA weights
    """
    print(f"Merging LoRA weights from {lora_path} into {lora_path}_merged")
    
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Merge weights
    model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained(f"{lora_path}_merged")
    tokenizer.save_pretrained(f"{lora_path}_merged")


def prepare_output_file(args: argparse.Namespace) -> str:
    """
    Prepare output file path and create directories.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to output file
    """
    # if args.output_file is not None:
    #     return args.output_file
        
    # Create output directory based on model name
    if args.peft:
        output_dir = os.path.join(args.output_dir, args.peft_path.replace("/", "-"))
    else:
        output_dir = os.path.join(args.output_dir, args.model_name.replace("/", "-"))
        
    # Add RAG info if applicable
    if args.eval_rag:
        output_dir += f"_rag{args.rag_top_k}"
    
    if args.ds_size is not None and args.ds_size > 0:
        output_dir += f"_evalds{args.ds_size}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename based on input filename
    output_filename = os.path.basename(args.input_file).replace(".json", "_eval.json")
    return os.path.join(output_dir, output_filename)


def prepare_evaluation_examples(data: Dict[str, Any], args: argparse.Namespace) -> List[Example]:
    """
    Prepare evaluation examples from the data.
    
    Args:
        data: Loaded data from input file
        args: Command line arguments
        
    Returns:
        List of evaluation examples
    """
    eval_examples = []
    
    for datum in data[:args.ds_size]:
        year = datum['year']
        month = datum['month']
        date = datum['date']
        case_id = datum['case_id']
        if args.year is not None and year != args.year:
            continue
        if args.month is not None and month != args.month:
            continue
        if args.date is not None and date != args.date:
            continue
        
        if not args.no_reliability and 'reliability' in datum['eval']:
            if 'masked_prompt' in datum['eval']['reliability'] and "[MASK]" in datum['eval']['reliability']['masked_prompt']:
                input_str = datum['eval']['reliability']['masked_prompt']
            else:
                input_str = PROMPT_TEMPLATE.format(input_str=datum['eval']['reliability']['prompt'])
            example = Example(
                input_str=input_str,
                expected_output=datum['eval']['reliability']['answer'],
                question=datum['eval']['reliability']['prompt'],
                fact=datum['fact'],
                date=date,
                type="reliability",
                case_id=case_id
            )
            eval_examples.append(example)
        if not args.no_generality and 'generality' in datum['eval']:
            if 'masked_prompt' in datum['eval']['generality'] and "[MASK]" in datum['eval']['generality']['masked_prompt']:
                input_str = datum['eval']['generality']['masked_prompt']
            else:
                input_str = PROMPT_TEMPLATE.format(input_str=datum['eval']['generality']['prompt'])
            example = Example(
                input_str=input_str,
                expected_output=datum['eval']['generality']['answer'],
                question=datum['eval']['generality']['prompt'],
                fact=datum['fact'],
                date=date,
                type="generality",
                case_id=case_id
            )
            eval_examples.append(example)
        if not args.no_paraphrase and 'paraphrase' in datum['eval']:
            if 'masked_prompt' in datum['eval']['paraphrase'] and "[MASK]" in datum['eval']['paraphrase']['masked_prompt']:
                input_str = datum['eval']['paraphrase']['masked_prompt']
            else:
                input_str = PROMPT_TEMPLATE.format(input_str=datum['eval']['paraphrase']['prompt'])
            example = Example(
                input_str=input_str,
                expected_output=datum['eval']['paraphrase']['answer'],
                question=datum['eval']['paraphrase']['prompt'],
                fact=datum['fact'],
                date=date,
                type="paraphrase",
                case_id=case_id
            )
            eval_examples.append(example)
        if not args.no_portability and 'portability' in datum['eval']:
            if 'masked_prompt' in datum['eval']['portability'] and "[MASK]" in datum['eval']['portability']['masked_prompt']:
                input_str = datum['eval']['portability']['masked_prompt']
            else:
                input_str = PROMPT_TEMPLATE.format(input_str=datum['eval']['portability']['prompt'])
            example = Example(
                input_str=input_str,
                expected_output=datum['eval']['portability']['answer'],
                question=datum['eval']['portability']['prompt'],
                fact=datum['fact'],
                date=date,
                type="portability",
                case_id=case_id
            )
            eval_examples.append(example)
        if not args.no_counterfactual and 'counterfactual' in datum['eval']:
            if 'masked_prompt' in datum['eval']['counterfactual'] and "[MASK]" in datum['eval']['counterfactual']['masked_prompt']:
                input_str = datum['eval']['counterfactual']['masked_prompt']
            else:
                input_str = PROMPT_TEMPLATE.format(input_str=datum['eval']['counterfactual']['prompt'])
            example = Example(
                input_str=input_str,
                expected_output=datum['eval']['counterfactual']['answer'],
                question=datum['eval']['counterfactual']['prompt'],
                fact=datum['fact'],
                date=date,
                type="counterfactual",
                case_id=case_id
            )
            eval_examples.append(example)
            if 'masked_prompt' in datum['eval']['factual'] and "[MASK]" in datum['eval']['factual']['masked_prompt']:
                input_str = datum['eval']['factual']['masked_prompt']
            else:
                input_str = PROMPT_TEMPLATE.format(input_str=datum['eval']['factual']['prompt'])
            example = Example(
                input_str=input_str,
                expected_output=datum['eval']['factual']['answer'],
                question=datum['eval']['factual']['prompt'],
                fact=datum['fact'],
                date=date,
                type="factual"
            )
            eval_examples.append(example)
        if not args.no_locality and 'locality' in datum['eval']:
            if 'masked_prompt' in datum['eval']['locality'] and "[MASK]" in datum['eval']['locality']['masked_prompt']:
                input_str = datum['eval']['locality']['masked_prompt']
            else:
                input_str = PROMPT_TEMPLATE.format(input_str=datum['eval']['locality']['prompt'])
            example = Example(
                input_str=input_str,
                expected_output=datum['eval']['locality']['answer'],
                question=datum['eval']['locality']['prompt'],
                fact=datum['fact'],
                date=date,
                type="locality",
                case_id=case_id
            )
            eval_examples.append(example)
    
    if args.predict_mask:
        for example in eval_examples:
            example.input_str = AR_MASK_PREDICT_PROMPT.format(input_str=example.input_str + " [MASK]")
    
    return eval_examples

def prepare_rag_context(
    eval_examples: List[Example], 
    embedder: SentenceTransformer, 
    datastore: List[str], 
    embeddings: torch.Tensor,
    rag_top_k: int
) -> List[Example]:
    """
    Prepare RAG context for evaluation examples.
    
    Args:
        eval_examples: List of evaluation examples
        embedder: Sentence transformer model
        datastore: List of document texts
        embeddings: Embeddings of all documents
        rag_top_k: Number of documents to retrieve
        
    Returns:
        Updated list of evaluation examples with RAG context
    """
    updated_examples = []
    
    for i, example in tqdm(enumerate(eval_examples), total=len(eval_examples), desc="Retrieving pages"):
        # Get query embedding
        query_embedding = embedder.encode(example.input_str, convert_to_tensor=True)
        
        # Retrieve relevant context
        retrieved_pages = retrieve_page(query_embedding, embeddings, datastore, k=rag_top_k)
        
        # Format context text
        context_txt = "\n\n".join([f"Context {i+1}. {page}" for i, page in enumerate(retrieved_pages)])
        
        # Create new example with RAG context
        updated_examples.append(Example(
            input_str=PROMPT_TEMPLATE_RAG.format(context=context_txt, input_str=example.question),
            expected_output=example.expected_output,
            question=example.question,
            fact=example.fact,
            date=example.date,
            type=example.type
        ))
    
    return updated_examples

def initialize_vllm_model(args: argparse.Namespace) -> LLM:
    """
    Initialize vLLM model based on arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Initialized vLLM model
    """
    model_params = {
        "model": args.model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.model_max_len,
    }
    
    # Add quantization if requested
    if args.quantization:
        model_params.update({
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
        })
    
    # Add max_num_seqs if provided
    if args.max_num_seqs is not None:
        model_params["max_num_seqs"] = args.max_num_seqs
    
    return LLM(**model_params)

def save_checkpoint(checkpoint_file: str, 
                   processed_examples: List[int], 
                   results: List[Result]) -> None:
    """
    Save checkpoint with processed examples and current results.
    
    Args:
        checkpoint_file: Path to checkpoint file
        processed_examples: List of indices of processed examples
        results: List of results obtained so far
    """
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    
    checkpoint_data = {
        "processed_examples": processed_examples,
        "results": results
    }
    
    with open(checkpoint_file, "w", encoding="utf-8") as file:
        json.dump(checkpoint_data, file)
    
    # print(f"Checkpoint saved to {checkpoint_file}")


def save_retrieval_checkpoint(file_path: str, eval_examples_with_context: List[Example]) -> None:
    """
    Save retrieval results to a separate checkpoint file.
    
    Args:
        file_path: Path to retrieval checkpoint file
        eval_examples_with_context: List of examples with retrieved context
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Extract the essential data to save
    checkpoint_data = []
    for example in eval_examples_with_context:
        # Store the full input string with RAG context - these will be truncated during evaluation
        checkpoint_data.append({
            "input_str": example.input_str,
            "expected_output": example.expected_output,
            "question": example.question,
            "fact": example.fact,
            "date": example.date,
            "type": example.type
        })
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(checkpoint_data, file)
    
    print(f"Retrieval checkpoint saved to {file_path}")


def load_retrieval_checkpoint(file_path: str) -> List[Example]:
    """
    Load retrieval results from a checkpoint file.
    
    Args:
        file_path: Path to retrieval checkpoint file
        
    Returns:
        List of examples with retrieved context
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            checkpoint_data = json.load(file)
            
        examples = []
        for data in checkpoint_data:
            example = Example(
                input_str=data["input_str"],
                expected_output=data["expected_output"],
                question=data["question"],
                fact=data["fact"],
                date=data["date"],
                type=data["type"]
            )
            examples.append(example)
        
        print(f"Loaded retrieval checkpoint from {file_path}")
        print(f"Loaded {len(examples)} examples with retrieved context")
        
        return examples
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Couldn't load retrieval checkpoint: {e}")
        return []


def get_retrieval_checkpoint_path(args: argparse.Namespace) -> str:
    """
    Get path for the retrieval checkpoint file.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to retrieval checkpoint file
    """
    # Create a base directory for retrieval checkpoints
    retrieval_dir = os.path.join(args.retrieval_output_dir)
    os.makedirs(retrieval_dir, exist_ok=True)
    
    # Create filename based on input file and RAG parameters
    base_name = os.path.basename(args.input_file).replace(".json", "")
    checkpoint_name = f"{base_name}_rag{args.rag_top_k}_retrieval.json"
    
    return os.path.join(retrieval_dir, checkpoint_name)


def load_checkpoint(checkpoint_file: str) -> Tuple[List[int], List[Result]]:
    """
    Load checkpoint with processed examples and current results.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple containing:
        - List of indices of processed examples
        - List of results obtained so far
    """
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as file:
            checkpoint_data = json.load(file)
            
        processed_examples = checkpoint_data.get("processed_examples", [])
        results = checkpoint_data.get("results", [])
        
        print(f"Loaded checkpoint from {checkpoint_file}")
        print(f"Resuming from {len(processed_examples)} processed examples with {len(results)} results")
        
        return processed_examples, results
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Couldn't load checkpoint: {e}")
        return [], []

def get_model_type(model_name):
    """Get the model type based on its name."""
    if "t5" in model_name:
        return "t5"
    elif "bert" in model_name:
        return "bert"
    else:
        return "ar"

def predict_with_t5(model, tokenizer, question, max_length=50):
    """
    Predict answer using T5 model with span format.
    
    Args:
        model: The T5 model
        tokenizer: The tokenizer
        question: Input question string (will add <extra_id_0> at the end)
        max_length: Maximum generation length
    
    Returns:
        str: Predicted answer
    """
    # Add sentinel token to mark where prediction should occur
    if "[MASK]" in question:
        input_text = question.replace("[MASK]", "<extra_id_0>")
    else:
        input_text = f"{question}<extra_id_0>"
    # input_text = "<extra_id_0> built both an island of trash and an island of hope"
    
    # Tokenize input text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate prediction
    with torch.no_grad():
        # outputs = model.generate(
        outputs = accelerator.unwrap_model(model).generate(
            **inputs,
            max_length=max_length,
        )
    
    # Decode prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # replace extra ids
    for i in range(10):
        prediction = prediction.replace(f"<extra_id_{i}>", "")
    
    # Clean up any extra spaces
    prediction = prediction.strip()
    
    return prediction

def predict_with_bert(model, tokenizer, text, num_masks=1, max_length=50):
    """
    Predict masked tokens using BERT-style masked language modeling.
    
    Args:
        model: The BERT model
        tokenizer: The tokenizer
        text: Input text string
        num_masks: Number of mask tokens to append at the end
        max_length: Maximum sequence length
    
    Returns:
        str: Text with predictions for masked tokens
    """
    # Append mask tokens to the end of the input text
    mask_token = tokenizer.mask_token
    input_text = f"{text} " + " ".join([mask_token] * num_masks)
    
    # Tokenize input text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Find positions of the mask tokens
    mask_token_indices = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    
    # Get model output
    with torch.no_grad():
        # outputs = model(**inputs)
        outputs = accelerator.unwrap_model(model)(**inputs)
        predictions = outputs.logits
    
    # Get top predictions for each mask position
    predicted_tokens = []
    for mask_idx in mask_token_indices:
        mask_predictions = predictions[0, mask_idx]
        top_token_id = torch.argmax(mask_predictions).item()
        predicted_token = tokenizer.convert_ids_to_tokens(top_token_id)
        predicted_tokens.append(predicted_token)
        
    # Clean up any extra spaces
    # result = " ".join(predicted_tokens)
    result = tokenizer.convert_tokens_to_string(predicted_tokens)
    
    return result.strip()

def run_t5_evaluation(
        args, eval_examples,
        checkpoint_file, checkpoint_frequency=50
):
    rank, world_sz = accelerator.process_index, accelerator.num_processes

    # --------- per‑rank checkpoint path & safe load --------- #
    rank_ckpt = checkpoint_file.replace(".json", f".rank{rank}.json")
    if os.path.exists(rank_ckpt):
        processed_indices, results = load_checkpoint(rank_ckpt)
    else:
        processed_indices, results = [], []

    # ---------- prepare model / tokenizer ---------- #
    model      = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer  = AutoTokenizer.from_pretrained(args.model_name)

    model, tokenizer = accelerator.prepare(model, tokenizer)
    print("model prepared")
    model.eval()

    # ---------- main evaluation loop ---------- #
    for idx, ex in tqdm(
            list(enumerate(eval_examples)),  # list() avoids tqdm/skip colliding
            desc=f"Rank {rank} evaluation", total=len(eval_examples)
    ):
        if idx % world_sz != rank:
            continue                 # not this rank’s responsibility
        if idx in processed_indices:
            continue                 # already done on this rank

        output = predict_with_t5(model, tokenizer, ex.input_str,
                                 max_length=args.max_new_tokens + 10)

        results.append({
            "input":     ex.input_str,
            "output":    output,
            "expected_output": ex.expected_output,
            "question":  ex.question,
            "fact":      ex.fact,
            "date":      ex.date,
            "type":      ex.type,
            "case_id":   ex.case_id,
            "correct":   compare(output, ex.expected_output),
            "global_idx": idx,
        })
        processed_indices.append(idx)

        # periodic flush
        if len(processed_indices) % checkpoint_frequency == 0:
            save_checkpoint(rank_ckpt, processed_indices, results)

    # final flush for this rank
    save_checkpoint(rank_ckpt, processed_indices, results)
    accelerator.wait_for_everyone()

    # ---------- merge (main process only) ---------- #
    global_results = results            # default = this rank’s shard
    if accelerator.is_main_process:
        merged_idx, merged_res = [], []
        shard_pattern = checkpoint_file.replace(".json", ".rank*.json")
        for fn in sorted(glob.glob(shard_pattern)):
            idxs, res = load_checkpoint(fn)
            merged_idx.extend(idxs)
            merged_res.extend(res)

        merged_idx = sorted(set(merged_idx))
        merged_res.sort(key=lambda r: r["global_idx"])

        # write a single global checkpoint (optional)
        save_checkpoint(checkpoint_file, merged_idx, merged_res)

        # clean up the per‑rank shards
        for fn in glob.glob(shard_pattern):
            os.remove(fn)

        global_results = merged_res     # <-- what we’ll return

        if args.peft:
            rmtree(f"{args.peft_path}_merged")

    accelerator.wait_for_everyone()
    return global_results if accelerator.is_main_process else []

def aggregate_results(results):
    """
    Aggregates results by type, date, month, and year, calculating accuracy metrics.
    
    Args:
        results: List of result dictionaries with keys including 'type', 'date', and 'correct'
                where 'correct' is a dictionary with 'match' (0 or 1) and 'f1' (float) values
        
    Returns:
        Dictionary with multi-level aggregation, including accuracy and F1 metrics
    """
    # Initialize aggregation dictionaries
    by_type = {}
    by_date = {}
    by_month = {}
    by_year = {}
    by_date_and_type = {}
    by_month_and_type = {}
    by_year_and_type = {}
    
    # Process each result
    for result in results:
        result_type = result.get('type')
        date_str = result.get('date')
        correct_dict = result.get('correct', {})
        
        # Skip if missing critical data
        if not result_type or not date_str or not isinstance(correct_dict, dict):
            continue
        
        # Extract match and f1 values
        is_match = correct_dict.get('match', 0)
        f1_score = correct_dict.get('f1', 0.0)
        
        # Parse date in format "31 January 2022"
        try:
            date_obj = datetime.strptime(date_str, "%d %B %Y")
            month_str = date_obj.strftime("%B %Y")
            year_str = str(date_obj.year)
        except ValueError:
            # Skip invalid dates
            continue
            
        # Aggregate by type
        if result_type not in by_type:
            by_type[result_type] = {'total': 0, 'matches': 0, 'f1_sum': 0.0}
        by_type[result_type]['total'] += 1
        by_type[result_type]['matches'] += is_match
        by_type[result_type]['f1_sum'] += f1_score
            
        # Aggregate by date
        if date_str not in by_date:
            by_date[date_str] = {'total': 0, 'matches': 0, 'f1_sum': 0.0}
        by_date[date_str]['total'] += 1
        by_date[date_str]['matches'] += is_match
        by_date[date_str]['f1_sum'] += f1_score
            
        # Aggregate by month
        if month_str not in by_month:
            by_month[month_str] = {'total': 0, 'matches': 0, 'f1_sum': 0.0}
        by_month[month_str]['total'] += 1
        by_month[month_str]['matches'] += is_match
        by_month[month_str]['f1_sum'] += f1_score
            
        # Aggregate by year
        if year_str not in by_year:
            by_year[year_str] = {'total': 0, 'matches': 0, 'f1_sum': 0.0}
        by_year[year_str]['total'] += 1
        by_year[year_str]['matches'] += is_match
        by_year[year_str]['f1_sum'] += f1_score
            
        # Aggregate by date and type
        if date_str not in by_date_and_type:
            by_date_and_type[date_str] = {}
        if result_type not in by_date_and_type[date_str]:
            by_date_and_type[date_str][result_type] = {'total': 0, 'matches': 0, 'f1_sum': 0.0}
        by_date_and_type[date_str][result_type]['total'] += 1
        by_date_and_type[date_str][result_type]['matches'] += is_match
        by_date_and_type[date_str][result_type]['f1_sum'] += f1_score
            
        # Aggregate by month and type
        if month_str not in by_month_and_type:
            by_month_and_type[month_str] = {}
        if result_type not in by_month_and_type[month_str]:
            by_month_and_type[month_str][result_type] = {'total': 0, 'matches': 0, 'f1_sum': 0.0}
        by_month_and_type[month_str][result_type]['total'] += 1
        by_month_and_type[month_str][result_type]['matches'] += is_match
        by_month_and_type[month_str][result_type]['f1_sum'] += f1_score
            
        # Aggregate by year and type
        if year_str not in by_year_and_type:
            by_year_and_type[year_str] = {}
        if result_type not in by_year_and_type[year_str]:
            by_year_and_type[year_str][result_type] = {'total': 0, 'matches': 0, 'f1_sum': 0.0}
        by_year_and_type[year_str][result_type]['total'] += 1
        by_year_and_type[year_str][result_type]['matches'] += is_match
        by_year_and_type[year_str][result_type]['f1_sum'] += f1_score
    
    # Calculate accuracy and average F1 percentages
    def add_metrics(stats_dict):
        for key, stats in stats_dict.items():
            if isinstance(stats, dict) and 'total' in stats and 'matches' in stats and 'f1_sum' in stats:
                if stats['total'] > 0:
                    stats['match_accuracy'] = (stats['matches'] / stats['total'] * 100)
                    stats['avg_f1'] = (stats['f1_sum'] / stats['total'])
                else:
                    stats['match_accuracy'] = 0
                    stats['avg_f1'] = 0
            elif isinstance(stats, dict):
                add_metrics(stats)
    
    # Add metrics to all aggregation dictionaries
    add_metrics(by_type)
    add_metrics(by_date)
    add_metrics(by_month)
    add_metrics(by_year)
    add_metrics(by_date_and_type)
    add_metrics(by_month_and_type)
    add_metrics(by_year_and_type)
    
    # Create overall statistics
    overall = {
        'total': sum(stats['total'] for stats in by_type.values()),
        'matches': sum(stats['matches'] for stats in by_type.values()),
        'f1_sum': sum(stats['f1_sum'] for stats in by_type.values()),
    }
    
    # Calculate overall metrics
    if overall['total'] > 0:
        overall['match_accuracy'] = (overall['matches'] / overall['total'] * 100)
        overall['avg_f1'] = (overall['f1_sum'] / overall['total'])
    else:
        overall['match_accuracy'] = 0
        overall['avg_f1'] = 0
        
    # Create combined aggregation
    aggregation = {
        'by_type': by_type,
        'by_date': by_date,
        'by_month': by_month,
        'by_year': by_year,
        'by_date_and_type': by_date_and_type,
        'by_month_and_type': by_month_and_type,
        'by_year_and_type': by_year_and_type,
        'overall': overall
    }
    
    return aggregation


def main(args: argparse.Namespace) -> None:
    """
    Main evaluation function.
    
    Args:
        args: Command line arguments
    """    
    # Prepare output file
    output_file = prepare_output_file(args)
    print(f"Output file: {output_file}")

    checkpoint_file = output_file.replace(".json", "_checkpoint.json")
    print(f"Checkpoint file: {checkpoint_file}")

    # Check if output file exists and handle overwrite
    if os.path.exists(output_file) and not args.overwrite:
        # If table not generated, then generate it
        for metric in ["match_accuracy", "avg_f1"]:
            table_generator(
                input_file=output_file,
                output_file=output_file.replace(".json", f"_{metric}.tex"),
                metric=metric,
            )
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite.")
        return
    
    # Delete checkpoint file if overwrite is requested
    if args.overwrite and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Removed existing checkpoint file {checkpoint_file} due to --overwrite flag")
        
    # Load data
    with open(args.input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
        
    # Prepare evaluation examples
    eval_examples = prepare_evaluation_examples(data, args)
    print(f"Prepared {len(eval_examples)} evaluation examples")
    
    # Add RAG context if requested
    if args.eval_rag:
        # Get retrieval checkpoint path
        retrieval_checkpoint_path = get_retrieval_checkpoint_path(args)
        
        # Try to load retrieval checkpoint first
        eval_examples_with_context = load_retrieval_checkpoint(retrieval_checkpoint_path)
        
        # If no checkpoint found or mismatch in example count, perform retrieval
        if not eval_examples_with_context or len(eval_examples_with_context) != len(eval_examples):
            print("Preparing RAG datastore...")
            datastore = []
            for datum in data:
                if 'bold_entity_page' in datum and datum['bold_entity_page']:
                    datastore.append(datum['bold_entity_page']['content'])
            
            print(f"Loading embedding model: {DEFAULT_EMBEDDING_MODEL}")
            embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            embeddings = embedder.encode(datastore, convert_to_tensor=True, show_progress_bar=True)
            print(f"Created embeddings for {len(datastore)} documents")
            
            # Perform retrieval and add context to examples
            eval_examples_with_context = prepare_rag_context(
                eval_examples, embedder, datastore, embeddings, args.rag_top_k
            )
            
            # Save retrieval checkpoint
            save_retrieval_checkpoint(retrieval_checkpoint_path, eval_examples_with_context)
        
        # Use the examples with context for evaluation
        eval_examples = eval_examples_with_context
    
    # Run evaluation with checkpointing
    checkpoint_frequency = 500
    
    # Run evaluation
    results = run_t5_evaluation(args, eval_examples, checkpoint_file, checkpoint_frequency)

    # ---------- only rank‑0 beyond this point ----------
    if not accelerator.is_main_process:
        # fix communication error
        # accelerator.wait_for_everyone()   # keep the other ranks alive until rank‑0 finishes
        return
    
    # Aggregate results
    aggregation = aggregate_results(results)
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump({
            "results": results,
            "aggregation": aggregation
        }, file, indent=4)
    
    # Generate table
    for metric in ["match_accuracy", "avg_f1"]:
        table_generator(
            input_file=output_file,
            output_file=output_file.replace(".json", f"_{metric}.tex"),
            metric=metric,
        )
    
    # delete checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Removed checkpoint file {checkpoint_file} after successful evaluation")


if __name__ == "__main__":
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Evaluate language models on question answering tasks")
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Name of the model to evaluate")
    model_group.add_argument("--peft", action="store_true",
                        help="Whether to use PEFT (Parameter-Efficient Fine-Tuning)")
    model_group.add_argument("--peft_path", type=str, 
                        default="train_results/wikidyk2022_questions_01082025_gpt-4o_eval-meta-llama_Llama-2-7b-chat-hf",
                        help="Path to PEFT weights")
    model_group.add_argument("--chat_mode", action="store_true",
                        help="Whether to use chat mode for inference")
    
    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--input_file", type=str, 
                       default="data/wikidyk2022_questions_12282024_gpt-4o-mini.json",
                       help="Path to input data file")
    data_group.add_argument("--output_dir", type=str, default="eval_results",
                       help="Directory to save output files")
    data_group.add_argument("--retrieval_output_dir", type=str, default="retrieval_results",
                       help="Directory to save retrieval checkpoint files")
    data_group.add_argument("--overwrite", action="store_true",
                       help="Whether to overwrite existing output file")
    data_group.add_argument("--ds_size", type=int, default=None,
                       help="Number of examples to evaluate (default: all)")
    
    # Evaluation options
    eval_group = parser.add_argument_group("Evaluation Options")
    eval_group.add_argument("--year", type=int, default=None,
                       help="Year to evaluate (default: all)")
    eval_group.add_argument("--month", type=int, default=None,
                       help="Month to evaluate (default: all)")
    eval_group.add_argument("--date", type=str, default=None,
                       help="Date to evaluate (default: all)")
    eval_group.add_argument("--no_reliability", action="store_true",
                       help="Skip reliability evaluation")
    eval_group.add_argument("--no_generality", action="store_true",
                       help="Skip generality evaluation")
    eval_group.add_argument("--no_paraphrase", action="store_true",
                       help="Skip paraphrase evaluation")
    eval_group.add_argument("--no_portability", action="store_true",
                       help="Skip portability evaluation")
    eval_group.add_argument("--no_counterfactual", action="store_true",
                       help="Skip counterfactual evaluation")
    eval_group.add_argument("--no_locality", action="store_true",
                       help="Skip locality evaluation")
    eval_group.add_argument("--predict_mask", action="store_true",
                       help="Predict masked tokens instead of span format")
    
    # RAG configuration
    rag_group = parser.add_argument_group("RAG Configuration")
    rag_group.add_argument("--eval_rag", action="store_true",
                      help="Whether to use RAG for evaluation")
    rag_group.add_argument("--rag_top_k", type=int, default=1,
                      help="Number of documents to retrieve for RAG")
    
    # Hardware configuration
    hw_group = parser.add_argument_group("Hardware Configuration")
    hw_group.add_argument("--tensor_parallel_size", type=int, default=1,
                     help="Number of GPUs to use for tensor parallelism")
    hw_group.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                     help="Fraction of GPU memory to use")
    hw_group.add_argument("--max_num_seqs", type=int, default=None,
                     help="Maximum number of sequences to process in parallel")
    
    # Model parameters
    params_group = parser.add_argument_group("Model Parameters")
    params_group.add_argument("--quantization", action="store_true",
                        help="Whether to use quantization")
    params_group.add_argument("--model_max_len", type=int, default=2048,
                        help="Maximum model sequence length")
    params_group.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    
    # Parse arguments and run main function
    args = parser.parse_args()
    
    # Set random seeds
    set_all_seeds()
    
    # Run main function
    main(args)