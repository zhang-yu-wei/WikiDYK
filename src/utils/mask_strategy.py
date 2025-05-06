import random
import torch
import math

def create_mlm_input(input_ids, tokenizer, mlm_probability=0.15, ignore_index=-100, max_span_length=5, seed=None):
    """
    Creates inputs for span-based masked language modeling by masking consecutive spans of tokens.
    
    Args:
        input_ids: Tensor of token ids
        tokenizer: Tokenizer instance with mask_token_id and vocab_size attributes
        mlm_probability: Target probability of masking a token (default: 0.15)
        ignore_index: Value used for unmasked positions in labels (default: -100)
        max_span_length: Maximum length of spans to mask (default: 5)
    
    Returns:
        tuple (masked_inputs, labels) where:
            - masked_inputs: Input sequence with masked spans
            - labels: Original tokens (ignore_index for unmasked positions)
    """
    if seed is not None:
        # Set temporary random seed
        old_state = random.getstate()
        random.seed(seed)
    
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
        
    labels = torch.full_like(input_ids, ignore_index)
    input_ids = input_ids.clone()
    
    # Create special tokens mask
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in tokenizer.all_special_ids:
        special_tokens_mask |= (input_ids == special_id)
    
    # Calculate number of spans needed to achieve mlm_probability
    seq_length = input_ids.size(0)
    avg_span_length = (max_span_length + 1) / 2  # Average span length
    num_tokens_to_mask = int(seq_length * mlm_probability)
    num_spans = math.ceil(num_tokens_to_mask / avg_span_length)
    
    # Create list of potential start positions (excluding special tokens)
    valid_start_positions = [
        i for i in range(seq_length - 1)
        if not special_tokens_mask[i] and not special_tokens_mask[i + 1]
    ]
    
    if not valid_start_positions:
        return input_ids, labels
    
    # Randomly select span start positions
    span_starts = sorted(random.sample(
        valid_start_positions,
        min(num_spans, len(valid_start_positions))
    ))
    
    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    
    # For each span start position
    for start_idx in span_starts:
        # Randomly select span length
        max_possible_length = min(
            max_span_length,
            seq_length - start_idx,
            next(
                (i for i in range(start_idx + 1, seq_length)
                 if special_tokens_mask[i]),
                seq_length
            ) - start_idx
        )
        span_length = random.randint(1, max_possible_length)
        
        # Mark indices in this span
        for i in range(start_idx, min(start_idx + span_length, seq_length)):
            if not special_tokens_mask[i]:
                masked_indices[i] = True
    
    # Set labels for masked tokens
    labels[masked_indices] = input_ids[masked_indices]
    
    # 80% of spans: replace with [MASK]
    indices_mask = torch.bernoulli(torch.full_like(input_ids, 0.8, dtype=torch.float32)).bool() & masked_indices
    input_ids[indices_mask] = tokenizer.mask_token_id
    
    # 10% of spans: replace with random tokens
    random_indices = torch.bernoulli(torch.full_like(input_ids, 0.5, dtype=torch.float32)).bool() & masked_indices & ~indices_mask
    random_tokens = torch.randint(
        0, tokenizer.vocab_size, size=(random_indices.sum(),), dtype=torch.long
    )
    input_ids[random_indices] = random_tokens
    
    # 10% of spans: keep original (no action needed)
    
    return input_ids, labels

def create_qa_mlm_input(input_ids, ans_input_ids, tokenizer, fixed_mask_length=10, ignore_index=-100, random_truncation=True, min_truncation_ratio=None):
    """
    Create MLM pair for QA data with fixed mask length
    and padding prediction after EOS. Add the masks after the question and let it predict the answer tokens and padding tokens.
    
    Args:
        input_ids: Tensor containing token IDs
        ans_input_ids: Tensor containing token IDs for QA data
        tokenizer: Tokenizer to use
        fixed_mask_length: Fixed number of masked tokens to be predicted
        ignore_index: Index to use for tokens that should not contribute to the loss
        
    Returns:
        Tuple of (masked_input_ids, mlm_labels)
    """
    input_length = len(input_ids)
    labels = torch.full_like(input_ids, ignore_index)
    
    # Ensure we don't exceed the input length
    fixed_mask_length = min(fixed_mask_length, input_length)
    
    # Create a copy of input_ids to modify
    masked_input_ids = input_ids.clone()
    
    # Find the end of the question
    # Typically would be where the answer starts or a special token like [SEP]
    # For simplicity, let's assume the answer starts right after the input
    question_end_idx = len(input_ids)
    
    # Calculate how many tokens to mask (minimum of fixed_mask_length or available answer tokens)
    ans_length = len(ans_input_ids)
    
    # Create mask tokens
    mask_token_id = tokenizer.mask_token_id
    
    # Add mask tokens after the question
    # Replace the original input_ids with mask tokens for the part we want to predict
    masked_input_ids = torch.cat([
        input_ids,
        torch.full((fixed_mask_length,), mask_token_id, dtype=input_ids.dtype, device=input_ids.device)
    ])
    
    # Create labels: set the masked positions to the answer token IDs
    # The labels tensor needs to be the same length as masked_input_ids
    labels = torch.cat([
        torch.full_like(input_ids, ignore_index),  # Original question tokens are ignored in loss
        ans_input_ids[:fixed_mask_length]  # Answer tokens that should be predicted
    ])
    
    # If the answer is shorter than fixed_mask_length, pad with EOS token
    if ans_length < fixed_mask_length:
        # Get EOS token ID
        eos_token_id = tokenizer.eos_token_id
        
        # Fill the remaining masked positions with EOS tokens in labels
        # These positions already have MASK tokens in masked_input_ids
        # labels[input_length + ans_length:input_length + fixed_mask_length] = eos_token_id
        labels = torch.cat([
            labels,
            torch.full((fixed_mask_length - ans_length,), eos_token_id, dtype=labels.dtype, device=labels.device)
        ])
    
    # Apply random truncation if enabled
    if random_truncation:        
        # Calculate minimum length to keep based on the ratio
        if min_truncation_ratio is None:
            min_length = 1
        else:
            min_length = int(fixed_mask_length * min_truncation_ratio)
        
        # Randomly choose a new length between min_length and total_length
        trunc_length = random.randint(min_length, fixed_mask_length)
        
        # Truncate both masked_input_ids and labels to the new length
        masked_input_ids = masked_input_ids[:-trunc_length]
        labels = labels[:-trunc_length]
    
    return masked_input_ids, labels

def generate_all_masked_versions(text, min_words=1, max_words=5, mask_token="<extra_id_0>"):
    """
    Generate all possible T5-style masked versions of the text.
    Each masked version has one contiguous span (of full words) masked, ensuring there's at least one word before and after.
    
    Args:
        text (str): The input text.
        min_words (int): Minimum number of words to mask.
        max_words (int): Maximum number of words to mask.
        
    Returns:
        List of tuples (input_text, target_text), each representing a unique masked variant.
    """
    words = text.split()
    candidates = []
    
    # Loop through possible start indices for masking.
    for start_idx in range(0, len(words) - min_words + 1):
        # For each start position, iterate over allowed span lengths.
        for span_length in range(min_words, max_words + 1):
            end_idx = start_idx + span_length
            # Ensure there's at least one word after the masked span.
            if end_idx < len(words) + 1:
                target_text = " ".join(words[start_idx:end_idx])
                masked_input = words[:start_idx] + [mask_token] + words[end_idx:]
                input_text = " ".join(masked_input)
                candidates.append((input_text, target_text))
    
    return candidates

def sample_masked_versions(text, num_samples, min_words=1, max_words=5, mask_token="<extra_id_0>"):
    """
    Sample a specified number of masked versions from all possible candidates.
    If the number of requested samples exceeds the number of unique candidates,
    the function repeats a random subset of the available ones to reach the total count.
    
    Args:
        text (str): The input text.
        num_samples (int): Desired number of masked versions.
        min_words (int): Minimum number of words to mask.
        max_words (int): Maximum number of words to mask.
    
    Returns:
        List of tuples (input_text, target_text), containing exactly num_samples entries.
    """
    candidates = generate_all_masked_versions(text, min_words, max_words, mask_token)
    total_candidates = len(candidates)
    
    if total_candidates == 0:
        return []
    
    # If we have enough candidates, sample without replacement.
    if num_samples <= total_candidates:
        return random.sample(candidates, num_samples)
    
    # Otherwise, repeat for the remaining samples.
    result = []
    while len(result) < num_samples:
        result.extend(random.sample(candidates, min(num_samples - len(result), total_candidates)))
    
    return result

if __name__ == "__main__":
    # Example usage:
    text = "The quick brown fox jumps over the lazy dog"
    unique_masked_versions = sample_masked_versions(text, num_samples=1000, min_words=1, max_words=3)
    breakpoint()

    for idx, (inp, tgt) in enumerate(unique_masked_versions):
        print(f"Variant {idx+1}:")
        print("Input:", inp)
        print("Target:", tgt)
        print()
