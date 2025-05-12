import os
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main(input_file, out_dir="figure"):
    # read data
    with open(input_file, 'r') as f:
        data = json.load(f)['results']

    string_index = []
    match_accuracy = []
    token_f1 = []
    for item in data:
        if item['type'] in ['reliability', 'generality']:
            # find the position of the first expected char in the fact string
            # find all positions where an expected_output entry appears
            positions = [
                item['fact'].find(candidate)
                for candidate in item['expected_output']
                if item['fact'].find(candidate) != -1
            ]
            # skip if none of them appear
            if not positions:
                continue
            idx = max(positions)
            string_index.append(idx/len(item['fact']))
            match_accuracy.append(item['correct']['match'])
            token_f1.append(item['correct']['f1'])
    
    # make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    # only keep the indices where match_accuracy == 1
    correct_indices = [
        idx for idx, m in zip(string_index, match_accuracy) if m == 1
    ]

    # switch to a density-normalized histogram
    sns.histplot(
        x=correct_indices,
        stat="density",    # normalize so area = 1
        bins=30,
        alpha=0.6
    )
    plt.title("Density of Correct Matches by Normalized String Index")
    plt.xlabel("Normalized String Index (position / length)")
    plt.ylabel("Density")

    # save
    out_path = os.path.join(out_dir, 'performance_by_string_index.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot match accuracy and token F1 vs. string index"
    )
    parser.add_argument('--input_file', help="Path to your JSON data file")
    parser.add_argument('--out_dir', default='figure',
                        help="Where to save the output plot")
    args = parser.parse_args()
    main(args.input_file, args.out_dir)
