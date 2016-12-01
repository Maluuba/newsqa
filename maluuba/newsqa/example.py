# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import itertools
import os

try:
    # Prefer a more specific path for when you run from the root of this repo
    # or if the root of the repo is in your path.
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

    # Dump the dataset to one file.
    newsqa_data.dump(path='combined-newsqa-data-v1.csv')

    print("Some answers:")
    for _, row in itertools.islice(newsqa_data.get_questions_and_answers().iterrows(), 10):
        print(row)
