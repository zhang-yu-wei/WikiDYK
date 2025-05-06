import torch
import random
import numpy as np
from tqdm import tqdm
import asyncio
import os
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)

def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across random, numpy, and pytorch.
    
    Args:
        seed: The random seed to use (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


async def get_openai_response(client, message, model_name):
    """Get a single response from OpenAI API."""
    completion = await client.chat.completions.create(
        model=model_name,
        messages=message,
    )
    return completion.choices[0].message.content


async def batch_process_openai_requests(client, messages, description, batch_size=20, model_name=None):
    """Process OpenAI API requests in batches with progress tracking."""
    results = []
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        tasks = [get_openai_response(client, msg, model_name) for msg in batch]
        
        with tqdm(total=len(tasks), desc=f"{description} (batch {i//batch_size + 1})", position=1, leave=False) as pbar:
            for future in asyncio.as_completed(tasks):
                result = await future
                results.append(result)
                pbar.update(1)
                
    return results


def openai_async_inference(
    messages: List[List[Dict[str, str]]], 
    tqdm_description: str = "Calling OpenAI API", 
    model_name: Optional[str] = None,
    batch_size: int = 20
) -> List[str]:
    """
    Get responses from OpenAI API using async calls for efficiency.
    
    Args:
        messages: List of message lists to send to the API
        tqdm_description: Description for progress bar
        model_name: Name of the model to use
        batch_size: Number of requests to process in parallel
        
    Returns:
        List of model responses
    """
    # Select the appropriate client based on the model
    if 'gpt' in model_name:
        client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    else:
        client = AsyncOpenAI(
            api_key=os.environ.get('TOGETHER_API_KEY'),
            base_url=os.environ.get('TOGETHER_BASE_URL'),
        )

    # Create batch processing coroutine
    async def process_all_batches():
        return await batch_process_openai_requests(client, messages, tqdm_description, batch_size)
    
    # Get event loop and run batch processing
    loop = asyncio.get_event_loop()
    completions = loop.run_until_complete(process_all_batches())
    
    # Return stripped responses
    return [completion.strip() for completion in completions]

def get_model_type(model_name_or_path):
    """Determine model type from model name"""
    model_lower = model_name_or_path.lower()
    if 'gpt' in model_lower or 'gemma' in model_lower or 'llama' in model_lower or 'qwen' in model_lower:
        return "gpt"
    elif 'bert' in model_lower:
        return "bert"
    elif 't5' in model_lower or 'bart' in model_lower:
        return "t5"
    else:
        return "unknown"

def load_model(model_name_or_path, cache_dir, use_flash_attention_2=True):
    """
    load different types of models
    """
    model_lower = model_name_or_path.lower()
    if 'gpt' in model_lower or \
        'opt' in model_lower or \
        'llama' in model_lower or \
        'qwen' in model_lower or \
        'mistral' in model_lower or \
        'gemma' in model_lower:
        # decoder-only models follow original implementations
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            use_flash_attention_2=use_flash_attention_2,
            torch_dtype=torch.bfloat16,
        )
    elif 'bert' in model_lower:
        # encoder models decode targets by masking all the outputs
        return AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
    elif 't5' in model_lower or 'bart' in model_lower:
        # encoder-decoder models use seq2seq training
        return AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
    else:
        raise NotImplementedError("The model is not implemented currently: {}".format(model_name_or_path))