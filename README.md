# WikiDYK: Evaluation

This is the official implementation of paper **Bidirectional LMs are Better Knowledge Memorizers? A Benchmark for Real-world Knowledge Injection** (paper coming soon).

## Official Links

[[WikiDYK Dataset]](https://huggingface.co/datasets/YWZBrandon/wikidyk)

<!-- This is the official code for the paper: **MemoryLLM: Towards Self-Updatable Large Language Models**.   
The model is open-sourced at https://huggingface.co/YuWangX/memoryllm-7b -->

## Release Notes
- [2025/05/16] 🔥 The dataset `wikidyk` has been uploaded to [wikidyk](https://huggingface.co/datasets/YWZBrandon/wikidyk).

## Getting Started

### Environment Setup
Install requirements.
```bash
git clone git@github.com:zhang-yu-wei/WikiDYK.git
cd WikiDYK
conda create --name wikidyk python=3.10 -y
conda activate wikidyk
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
```

### Downloading Data
In order to reproduce the experiments, first download the datasets. Notice that we have prepared a ready-to-run dataset on huggingface which is basically a reformat of the original data with the same content.
```bash
huggingface-cli download YWZBrandon/wikidyk_exp --repo-type dataset --local-dir data
```

### Training

#### NTP Training
```bash
bash scripts/train_ar_full_all_8gpu.sh
```

#### QA Training
```bash
bash scripts/train_synthetic_qa_all_8gpu.sh
```

#### SP Training for CLMs
```bash
bash scripts/train_pred_mask_full_all_8gpu.sh
```

#### SP Training for BiLMs
```bash
bash scripts/train_full_flan_t5.sh
```

### Evaluation

#### Static Analysis
```bash
bash scripts/eval_qa_static.sh
```

#### RAG
```bash
bash scripts/eval_qa_rag.sh
```

#### Evaluation on Trained Models
For CLMs, use the following script for evaluation.
```bash
bash scripts/eval_qa_trained.sh
```
For BiLMs, use the following script for evaluation.
```bash
bash scripts/eval_qa_accelerate.sh
```

### Ensemble Model
To train scope classifier, use the following script
```
bash scripts/train_scope_clf_deberta_large.sh
```