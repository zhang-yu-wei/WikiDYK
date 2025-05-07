import json
import os
import re
import uuid
import argparse
from copy import deepcopy
from collections import defaultdict
from datasets import Dataset, DatasetDict
import random
random.seed(42)
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'

"""
format scope classification data:
each cluster is a scope. There are two types of clusters:
1. semantic clusters
2. temporal clusters
"""

month_map = {
    'January': '01',
    'February': '02',
    'March': '03',
    'April': '04',
    'May': '05',
    'June': '06',
    'July': '07',
    'August': '08',
    'September': '09',
    'October': '10',
    'November': '11',
    'December': '12'
}

def format_date(date_str):
    day, month, year = date_str.split()
    month = month_map[month]

    return f"{int(year):04d}-{month}-{int(day):02d}"

def cluster_facts(facts, n_clusters=10):
    # Load pre-trained sentence transformer model
    embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

    # Compute embeddings for facts
    embeddings = embedder.encode(facts)

    # Cluster embeddings
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)

    # return cluster index of each fact
    return labels

def group_facts_by_semantic(input_data, n_clusters=10):    
    # Extract facts from input data
    facts = [item['fact'] for item in input_data]

    # Cluster facts
    cluster_labels = cluster_facts(facts, n_clusters)

    # Group facts by cluster labels
    clustered_data = {}
    for i, label in enumerate(cluster_labels):
        if label not in clustered_data:
            clustered_data[label] = []
        clustered_data[label].append(input_data[i])
    
    return clustered_data

def group_facts_by_temporal(input_data, n_clusters=10):
    # format dates and then sort by date
    for item in input_data:
        item['date'] = format_date(item['date'])
    input_data.sort(key=lambda x: x['date'])

    # bin the data into n_clusters
    n = len(input_data)
    bin_size = n // n_clusters
    clustered_data = {}
    for i in range(n_clusters):
        start = i * bin_size
        end = (i + 1) * bin_size if i != n_clusters - 1 else n
        clustered_data[i] = input_data[start:end]
    return clustered_data

def main(args):
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    if args.n_clusters == 1:
        # if n_clusters is 1, just return the data as is
        clustered_data = {0: data}
    else:
        if args.cluster_type == 'semantic':
            clustered_data = group_facts_by_semantic(data, args.n_clusters)
        
        elif args.cluster_type == 'temporal':
            clustered_data = group_facts_by_temporal(data, args.n_clusters)
    
    # include additional negative data
    with open(args.additional_negative_data_path, 'r') as f:
        additional_negative_data = json.load(f)
    
    # save the clustered data into huggingface dataset format
    # also save into json format so that it can be used for training t5 models

    # first, unpack the data
    unpacked_data = []
    unpacked_data_json = defaultdict(list)
    for cluster_id, facts in clustered_data.items():
        for fact in facts:
            unpacked_data.append({
                'cluster_id': int(cluster_id),
                'text': fact['fact'],
                'case_id': fact['case_id'],
                'date': fact['date'],
            })
            unpacked_data_json[cluster_id].append(deepcopy(fact))
    # add additional negative data
    for fact in additional_negative_data:
        unpacked_data.append({
            'cluster_id': -1,
            'text': fact['text'],
            'case_id': 'neg_' + str(uuid.uuid4()),
            'date': fact['date'],
        })
    
    # format eval data
    eval_data = []
    for cluster_id, facts in clustered_data.items():
        for fact in facts:
            for tp in ['reliability', 'paraphrase', 'generality', 'portability', 'counterfactual']:
                if random.random() < 0.01 and tp in fact['eval']:
                    eval_data.append({
                        'cluster_id': int(cluster_id),
                        'text': fact['eval'][tp]['prompt'],
                        'case_id': fact['case_id'],
                        'date': fact['date'],
                    })
            for tp in ['locality']:
                if random.random() < 0.05 and tp in fact['eval']:
                    eval_data.append({
                        'cluster_id': -1,
                        'text': fact['eval'][tp]['prompt'],
                        'case_id': fact['case_id'],
                        'date': fact['date'],
                    })

    # then, convert to huggingface dataset format
    train_ds = Dataset.from_list(unpacked_data)
    eval_ds  = Dataset.from_list(eval_data)

    dataset = DatasetDict({
        "train": train_ds,
        "eval":  eval_ds,
    })
    # save the dataset
    if args.n_clusters == 1:
        dataset.save_to_disk(f"data/scope_clf_data/1_cluster")
    else:
        dataset.save_to_disk(f"data/scope_clf_data/{args.cluster_type}_{args.n_clusters}_clusters")
        for cluster_id in unpacked_data_json:
            with open(f"data/scope_clf_data/{args.cluster_type}_{args.n_clusters}_clusters_{cluster_id}.json", 'w') as f:
                json.dump(unpacked_data_json[cluster_id], f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Format data for scope classification")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--cluster_type', type=str, required=True, choices=['semantic', 'temporal'], help='Type of cluster to format')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters to create')
    parser.add_argument('--additional_negative_data_path', type=str, help='Path to the additional negative data file')
    args = parser.parse_args()
    main(args)