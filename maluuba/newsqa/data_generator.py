# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import logging
import os

from simplify import simplify
from split_dataset import split_data
from tokenize_dataset import tokenize

try:
    # Prefer a more specific path for when you run from the root of this repo
    # or if the root of the repo is in your Python path.
    from maluuba.newsqa.data_processing import NewsQaDataset
except:
    # In case you're running this file from this folder.
    from data_processing import NewsQaDataset

if __name__ == "__main__":
    dir_name = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_stories_path', default=os.path.join(dir_name, 'cnn_stories.tgz'),
                        help="The path to the CNN stories (cnn_stories.tgz).")
    parser.add_argument('--dataset_path', default=os.path.join(dir_name, 'newsqa-data-v1.csv'),
                        help="The path to the dataset with questions and answers.")
    args = parser.parse_args()

    newsqa_data = NewsQaDataset(args.cnn_stories_path, args.dataset_path)

    logger = logging.getLogger('newsqa')
    logger.setLevel(logging.INFO)

    # Dump the dataset to one file.
    newsqa_data.dump(path='combined-newsqa-data-v1.json')
    newsqa_data.dump(path='combined-newsqa-data-v1.csv')

    tokenized_data_path = os.path.join(dir_name, 'newsqa-data-tokenized-v1.csv')
    tokenize(output_path=tokenized_data_path)
    split_data(dataset_path=tokenized_data_path)
    simplify(output_dir_path='split_data')
