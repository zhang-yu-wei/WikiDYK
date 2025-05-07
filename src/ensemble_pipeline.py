"""
Notice that the new version does not need to load the models in the ensemble pipeline.
The ensemble pipeline is a simple pipeline that combines the predictions from multiple models.
1. Classifier to classify the facts to select models for prediction.
2. Predict answer from each model.
3. Combine the predictions from all models.
"""

import re
import os
import json
import torch
import random
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any, List

from utils.metrics import compare


def get_cluster_id_from_path(file_path):
    pattern = r"cluster_(\d+)"
    match = re.search(pattern, file_path)
    if match:
        return match.group(1)
    else:
        return None

# Type definitions for clarity
class Example:
    def __init__(self, input_str: str, expected_output: str, question: str, fact: str, date: str, type: str, case_id: str = None):
        self.input_str = input_str
        self.expected_output = expected_output
        self.question = question
        self.fact = fact
        self.date = date
        self.type = type
        self.case_id = case_id

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
        if args.year is not None and year != args.year:
            continue
        if args.month is not None and month != args.month:
            continue
        if args.date is not None and date != args.date:
            continue
        
        if not args.no_reliability:
            example = Example(
                input_str=PROMPT_TEMPLATE.format(input_str=datum['eval']['reliability']['prompt']),
                expected_output=datum['eval']['reliability']['answer'],
                question=datum['eval']['reliability']['prompt'],
                fact=datum['fact'],
                date=date,
                type="reliability",
                case_id=datum['case_id']
            )
            eval_examples.append(example)
        if not args.no_generality:
            example = Example(
                input_str=PROMPT_TEMPLATE.format(input_str=datum['eval']['generality']['prompt']),
                expected_output=datum['eval']['generality']['answer'],
                question=datum['eval']['generality']['prompt'],
                fact=datum['fact'],
                date=date,
                type="generality",
                case_id=datum['case_id']
            )
            eval_examples.append(example)
        if not args.no_paraphrase:
            example = Example(
                input_str=PROMPT_TEMPLATE.format(input_str=datum['eval']['paraphrase']['prompt']),
                expected_output=datum['eval']['paraphrase']['answer'],
                question=datum['eval']['paraphrase']['prompt'],
                fact=datum['fact'],
                date=date,
                type="paraphrase",
                case_id=datum['case_id']
            )
            eval_examples.append(example)
        if not args.no_portability:
            if 'portability' not in datum['eval']:
                continue
            example = Example(
                input_str=PROMPT_TEMPLATE.format(input_str=datum['eval']['portability']['prompt']),
                expected_output=datum['eval']['portability']['answer'],
                question=datum['eval']['portability']['prompt'],
                fact=datum['fact'],
                date=date,
                type="portability",
                case_id=datum['case_id']
            )
            eval_examples.append(example)
        if not args.no_counterfactual:
            example = Example(
                input_str=PROMPT_TEMPLATE.format(input_str=datum['eval']['counterfactual']['prompt']),
                expected_output=datum['eval']['counterfactual']['answer'],
                question=datum['eval']['counterfactual']['prompt'],
                fact=datum['fact'],
                date=date,
                type="counterfactual",
                case_id=datum['case_id']
            )
            eval_examples.append(example)
            example = Example(
                input_str=PROMPT_TEMPLATE.format(input_str=datum['eval']['factual']['prompt']),
                expected_output=datum['eval']['factual']['answer'],
                question=datum['eval']['factual']['prompt'],
                fact=datum['fact'],
                date=date,
                type="factual",
                case_id=datum['case_id']
            )
            eval_examples.append(example)
    
    if args.predict_mask:
        for example in eval_examples:
            example.input_str = AR_MASK_PREDICT_PROMPT.format(input_str=example.input_str + " [MASK]")
    
    return eval_examples


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

# Main ensemble pipeline
class EnsemblePipeline:
    def __init__(self, 
                 classifier_model_path,
                 ensemble_model_path,
                 ensemble_model_name):
        """
        Initialize the ensemble pipeline.
        """
        self.scope_classifier = RobertaMultiLabelClassifier.from_pretrained(classifier_model_path)
        # set to cuda and eval mode
        self.scope_classifier.to("cuda")
        self.scope_classifier.eval()
        # load ensemble models
        ensemble_model_path_strs = {
            str(i): ensemble_model_path.format(i, i) for i in range(self.scope_classifier.num_labels)
        }
        self.ensemble_model_type = get_model_type(ensemble_model_name)
        if self.ensemble_model_type == "t5":
            self.ensemble_models = {
                cluster_id: AutoModelForSeq2SeqLM.from_pretrained(model_path) for cluster_id, model_path in ensemble_model_path_strs.items()
            }
        elif self.ensemble_model_type == "bert":
            self.ensemble_models = {
                cluster_id: AutoModelForMaskedLM.from_pretrained(model_path) for cluster_id, model_path in ensemble_model_path_strs.items()
            }
        else:
            raise ValueError(f"Unsupported model type: {self.ensemble_model_type}")
        # set models to cuda and eval mode
        for model in self.ensemble_models.values():
            model.to("cuda")
            model.eval()
        self.tokenizer_cls = AutoTokenizer.from_pretrained(classifier_model_path)
        self.tokenizer_ens = AutoTokenizer.from_pretrained(ensemble_model_name)
    
    def answer_w_max(self, question):
        """
        Get the answer from the maximum model.
        """
        tokenized_question = self.tokenizer_cls(
            question,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        tokenized_question = {k: v.to("cuda") for k, v in tokenized_question.items()}
        cluster_id_logits = self.scope_classifier(input_ids=tokenized_question["input_ids"], attention_mask=tokenized_question["attention_mask"])
        cluster_id_predicted = torch.argmax(cluster_id_logits, dim=1).item()
        # print(f"Predicted cluster id: {cluster_id_predicted}")
        # print the probabilities
        cluster_id_probs = torch.softmax(cluster_id_logits, dim=1).squeeze().tolist()
        # round the numbers and show the cluster id for clarity
        cluster_id_probs = [round(prob, 4) for prob in cluster_id_probs]
        cluster_id_probs = {str(i): prob for i, prob in enumerate(cluster_id_probs)}
        # print(f"Cluster id probabilities: {cluster_id_probs}")

        answer_model = self.ensemble_models[str(int(cluster_id_predicted))]

        if self.ensemble_model_type == "t5":
            answer = predict_with_t5(answer_model, self.tokenizer_ens, question)
        elif self.ensemble_model_type == "bert":
            answer = predict_with_bert(answer_model, self.tokenizer_ens, question)
        else:
            raise ValueError(f"Unsupported model type: {self.ensemble_model_type}")
        return answer, cluster_id_predicted, cluster_id_probs

    def answer_w_avg(self, question):
        """
        Get the answer from the all model predictions and select the best one.
        """
        raise NotImplementedError("Average prediction is not implemented yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Pipeline for Model Predictions")

    # ===== classifier model =====
    parser.add_argument("--classifier_model_path", type=str, required=True, help="Path to the classifier model")
    parser.add_argument("--classifier_model_name", type=str, default="roberta-large", help="Name of the classifier model")

    # ===== ensemble model =====
    parser.add_argument("--ens_model_path", type=str, required=True, help="Path to the ensemble models")
    parser.add_argument("--ens_model_name", type=str, default="roberta-large", help="Name of the ensemble model")
    parser.add_argument("--emb_name", type=str, default="bge-large-en-v1.5", help="Name of the embedding model")
    parser.add_argument("--num_models", type=int, default=10, help="Number of models in the ensemble")

    # ===== data =====
    parser.add_argument("--input_file", type=str, default="data/wikidyk2022-2024_01082025_gpt-4o_evalv2_pages_formatted_combined.json", help="Path to the input data file")
    parser.add_argument("--classification_labels", type=str, default=None, help="Path to additional classification data")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save the output")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists")
    parser.add_argument("--ds_size", type=int, default=None, help="Size of the dataset to use")

    # ===== evaluation settings =====
    parser.add_argument("--year", type=int, default=None,
                       help="Year to evaluate (default: all)")
    parser.add_argument("--month", type=int, default=None,
                       help="Month to evaluate (default: all)")
    parser.add_argument("--date", type=str, default=None,
                       help="Date to evaluate (default: all)")
    parser.add_argument("--no_reliability", action="store_true",
                       help="Skip reliability evaluation")
    parser.add_argument("--no_generality", action="store_true",
                       help="Skip generality evaluation")
    parser.add_argument("--no_paraphrase", action="store_true",
                       help="Skip paraphrase evaluation")
    parser.add_argument("--no_portability", action="store_true",
                       help="Skip portability evaluation")
    parser.add_argument("--no_counterfactual", action="store_true",
                       help="Skip counterfactual evaluation")
    parser.add_argument("--predict_mask", action="store_true",
                       help="Predict masked tokens instead of span format")

    args = parser.parse_args()

    # Load the model
    ens_pipeline = EnsemblePipeline(
        classifier_results
        prediction_results
    )
    # Load the data
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Prepare the evaluation examples
    eval_examples = prepare_evaluation_examples(data, args)

    # Modify the output name
    ens_name_in_path = "-".join(args.ens_model_name.split("/")[1:])
    classifier_name_in_path = "-".join(args.classifier_model_name.split("/")[1:])
    output_name = os.path.join(args.output_dir, f"cls={classifier_name_in_path}_ens={ens_name_in_path}_emb={args.emb_name}_eval_results.json")

    results = []
    for i, example in tqdm(enumerate(eval_examples), total=len(eval_examples)):
        # Get the answer from the ensemble pipeline
        answer, cluster_id_predicted, cluster_id_probs = ens_pipeline.answer_w_max(example.input_str)
        # Print the result
        # print(f"Input: {example.input_str}")
        # print(f"Expected Output: {example.expected_output}")
        # print(f"Predicted Output: {answer}")
        # print(f"Predicted Cluster ID: {cluster_id_predicted}")
        # print(f"Ground Truth Cluster ID: {classification_labels[example.case_id] if classification_labels else None}")
        # print("=" * 50)

        results.append({
            "input": example.input_str,
            "expected_output": example.expected_output,
            "output": answer,
            "question": example.question,
            "fact": example.fact,
            "date": example.date,
            "type": example.type,
            "case_id": example.case_id,
            "cluster_id_predicted": cluster_id_predicted,
            "cluster_id_probs": {cid: float(prob) for cid, prob in cluster_id_probs.items()},
            "cluster_id_ground_truth": classification_labels[example.case_id] if classification_labels else None,
            "correct": compare(answer, example.expected_output),
        })

        # compute average classification accuracy, average f1 and match accuracy
        if i % 100 == 0:
            classification_accuracy = 0
            f1 = 0
            match_accuracy = 0
            for result in results:
                if int(result["cluster_id_ground_truth"]) == int(result["cluster_id_predicted"]):
                    classification_accuracy += 1
                f1 += result["correct"]['f1']
                match_accuracy += result["correct"]['match']
            classification_accuracy /= len(results)
            f1 /= len(results)
            match_accuracy /= len(results)
            print(f"Classification Accuracy: {classification_accuracy}")
            print(f"F1: {f1}")
            print(f"Match Accuracy: {match_accuracy}")
    
        # Save the result
        if i % 100 == 0:
            aggregation = aggregate_results(results)
            with open(output_name, 'w') as f:
                json.dump({
                    "results": results,
                    "aggregation": aggregation
                }, f, indent=4)
    
    with open(output_name, 'w') as f:
        json.dump({
            "results": results,
            "aggregation": aggregate_results(results)
        }, f, indent=4)