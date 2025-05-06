#!/usr/bin/env python3
"""
LaTeX Table Generator Script for Evaluation Results

This script processes the evaluation results from the language model evaluation
and generates a LaTeX table showing performance broken down by month and evaluation type.
"""

import argparse
import json
import os
import csv
from datetime import datetime
from collections import defaultdict


def load_evaluation_results(file_path):
    """
    Load evaluation results from JSON file.
    
    Args:
        file_path: Path to the evaluation results JSON file
        
    Returns:
        Dictionary containing the evaluation results or None if file cannot be loaded
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def extract_monthly_metrics(results, metric_key='match_accuracy'):
    """
    Extract monthly metrics from evaluation results, organized by year.
    
    Args:
        results: Evaluation results dictionary
        metric_key: The metric to extract (e.g., 'match_accuracy', 'avg_f1')
        
    Returns:
        Dictionary with years, evaluation types, and monthly metrics
    """
    # Get monthly aggregated results
    monthly_data = results.get('aggregation', {}).get('by_month_and_type', {})
    
    # Initialize data structure
    # Format: {year: {eval_type: {month: value}}}
    yearly_monthly_metrics = defaultdict(lambda: defaultdict(dict))
    
    # Process each month's data
    for month_str, type_data in monthly_data.items():
        try:
            # Parse month string (e.g., "January 2022")
            month_date = datetime.strptime(month_str, "%B %Y")
            year = month_date.year
            month_num = month_date.month  # 1-12
            
            # Extract metrics for each evaluation type
            for eval_type, metrics in type_data.items():
                if metric_key in metrics:
                    yearly_monthly_metrics[year][eval_type][month_num] = metrics[metric_key]
        except ValueError:
            # Skip if month format is unexpected
            continue
    
    return yearly_monthly_metrics


def generate_latex_table(yearly_monthly_metrics, model_name, output_file):
    """
    Generate LaTeX table from monthly metrics for each year and save to file.
    
    Args:
        yearly_monthly_metrics: Dictionary with years, evaluation types, and monthly metrics
        model_name: Name of the model to include in the table
        output_file: Path to save the LaTeX table
    """
    # Define evaluation types in desired order
    eval_types = ['reliability', 'generality', 'paraphrase', 'portability', 'counterfactual', 'factual', 'locality']
    
    # Start building the LaTeX table content
    all_tables = []
    
    # Sort years to process them in chronological order
    years = sorted(yearly_monthly_metrics.keys())
    
    if not years:
        raise ValueError("No data found to generate tables")
    
    for year in years:
        monthly_metrics = yearly_monthly_metrics[year]
        
        # Skip if no data for this year
        if not monthly_metrics:
            continue
            
        table_lines = []
        
        # Add year as a section header
        table_lines.append(f"% Results for year {year}")
        table_lines.append(r"\subsection*{" + f"Results for {year}" + r"}")
        
        # Table header
        table_lines.append(r"\begin{table}[ht]")
        table_lines.append(r"\centering")
        table_lines.append(r"\begin{tabular}{l|l|*{12}{c}}")
        table_lines.append(r"\hline")
        table_lines.append(r"Model & Type & 01 & 02 & 03 & 04 & 05 & 06 & 07 & 08 & 09 & 10 & 11 & 12 \\")
        table_lines.append(r"\midrule")
        
        # Flag to track if we've inserted the model name
        model_name_inserted = False
        
        # Add rows for each evaluation type
        for i, eval_type in enumerate(eval_types):
            if eval_type in monthly_metrics:
                # Insert model name before paraphrase (assume paraphrase is 3rd in the list)
                if eval_type == 'paraphrase' and not model_name_inserted:
                    row = f"{model_name} & {eval_type.capitalize()} & "
                    model_name_inserted = True
                else:
                    row = f" & {eval_type.capitalize()} & "
                
                # Add monthly values
                month_values = []
                for month in range(1, 13):  # 1-12
                    if month in monthly_metrics[eval_type]:
                        # Format to 2 decimal places
                        value = f"{monthly_metrics[eval_type][month]:.2f}"
                        month_values.append(value)
                    else:
                        # Empty if no data for this month
                        month_values.append(" ")
                
                # Join values and complete the row
                row += " & ".join(month_values) + r" \\"
                table_lines.append(row)
        
        # If we never inserted the model name, insert it at the first row with data
        if not model_name_inserted and table_lines[-1].startswith(" &"):
            # Replace the first data row to include the model name
            table_lines[-1] = model_name + table_lines[-1][1:]
        
        # Table footer
        table_lines.append(r"\hline")
        table_lines.append(r"\end{tabular}")
        table_lines.append(r"\caption{Monthly evaluation results for " + model_name + f" ({year})" + r"}")
        table_lines.append(r"\end{table}")
        table_lines.append("")  # Add a blank line between tables
        
        all_tables.extend(table_lines)
    
    # Write to file
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("\n".join(all_tables))
    except Exception as e:
        raise IOError(f"Failed to write to output file '{output_file}': {str(e)}")

    # print output tables
    for line in all_tables:
        print(line)
    
    return len(years)  # Return number of tables generated

def generate_table_by_type(metrics_by_type, model_name, output_file, metric_name):
    """
    Generate a table of metrics by evaluation type, saving both .csv and .tex.

    Args:
        metrics_by_type: dict mapping eval_type -> { metric_name: float, … }
        model_name:       str, name of the model (will appear in every row)
        output_file:      str, path to write one of the files. If it ends
                          in .csv, the .tex will be side-by-side (and vice versa);
                          otherwise both will be written as output_file+".csv"
                          and output_file+".tex".
    """
    # 1) Derive csv_file / tex_file paths
    base, ext = os.path.splitext(output_file)
    if ext.lower() == '.csv':
        csv_file = output_file
        tex_file = base + '.tex'
    elif ext.lower() == '.tex':
        tex_file = output_file
        csv_file = base + '.csv'
    else:
        csv_file = output_file + '.csv'
        tex_file = output_file + '.tex'
    
    eval_types = ['reliability', 'generality', 'paraphrase', 'portability', 'counterfactual', 'factual', 'locality']

    # 2) Gather all metric names, sort for stable column order
    all_metrics = {}
    for eval_type in eval_types:
        all_metrics.update({eval_type: metrics_by_type.get(eval_type, {}).get(metric_name, None)})

    # 3) Write CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        # first row: eval question type
        writer.writerow(eval_types)
        # one row for the metric values
        writer.writerow([all_metrics.get(eval_type, None) for eval_type in eval_types])
    
    # 4) Build LaTeX table lines
    table_lines = []

    header_line = r""
    for eval_type in eval_types:
        header_line += f" & {eval_type}"
    value_line = r""
    for eval_type in eval_types:
        value = all_metrics.get(eval_type, None)
        if value is not None:
            value_line += f" & {value:.2f}"
        else:
            value_line += " & "
    table_lines.append(header_line + r" \\")
    table_lines.append(value_line + r" \\")

    # 5) Write to .tex
    with open(tex_file, 'w', encoding='utf-8') as f_tex:
        for line in table_lines:
            f_tex.write(line + "\n")
    print(f"✔ LaTeX saved to {tex_file}")

def table_generator(input_file, output_file="evaluation_table.tex", model_name="", metric="match_accuracy", year=None):
    
    # Validate file paths
    if not os.path.exists(os.path.dirname(output_file)) and os.path.dirname(output_file):
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            print(f"Created directory: {os.path.dirname(output_file)}")
        except Exception as e:
            print(f"Error creating output directory: {str(e)}")
            return
    
    # Load evaluation results
    results = load_evaluation_results(input_file)
    if results is None:
        return  # Exit if file couldn't be loaded
    
    # Extract yearly monthly metrics
    yearly_monthly_metrics = extract_monthly_metrics(results, metric)
    
    # Check if any data was found
    if not yearly_monthly_metrics:
        print(f"No evaluation data found in the file using metric '{metric}'.")
        return
    
    # Filter for specific year if requested
    if year is not None:
        if year in yearly_monthly_metrics:
            yearly_monthly_metrics = {year: yearly_monthly_metrics[year]}
        else:
            print(f"Warning: No data found for year {year}")
            return
    
    # Generate and save LaTeX table
    try:
        generate_latex_table(yearly_monthly_metrics, model_name, output_file)
        print(f"Successfully generated LaTeX table at: {output_file}")
    except Exception as e:
        print(f"Error generating LaTeX table: {str(e)}")
        return

    metrics_by_type = results.get('aggregation', {}).get('by_type', {})
    generate_table_by_type(
        metrics_by_type=metrics_by_type,
        model_name=model_name,
        output_file=output_file.replace('.tex', '_by_type.csv'),
        metric_name=metric
    )
    generate_table_by_type(
        metrics_by_type=metrics_by_type,
        model_name=model_name,
        output_file=output_file.replace('.tex', '_by_type.tex'),
        metric_name=metric
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX table from evaluation results.")
    parser.add_argument("--input_file", help="Path to the input JSON file with evaluation results.")
    args = parser.parse_args()

    for metric in ['match_accuracy', 'avg_f1']:
        table_generator(
            input_file=args.input_file,
            output_file=os.path.splitext(args.input_file)[0] + f"_{metric}.tex",
            metric=metric
        )