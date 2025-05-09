# WikiDYKEvalV2

## Install
```bash
python -m pip install -r requirements.txt
wandb login
huggingface-cli login
git clone git@github.com:zhang-yu-wei/WikiDYKEvalV2.git
huggingface-cli download YWZBrandon/wikidyk --repo-type dataset --local-dir data
```

## run full ar training
```bash
bash scripts/train_synthetic_qa_all_8gpu.sh
```

## eval full ar training
Use `eval_qa.sh` for evaluation. Copy the dir name to `MODELS` argument. The script will automatically infer if it is peft. The copy the results from the .csv file in `eval_results` to this file: https://docs.google.com/spreadsheets/d/1COtdI3peeIYvD4qfrXoAa3jt3OUlqPoafzAGKjlZXsU/edit?usp=sharing