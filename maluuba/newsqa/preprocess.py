import logging
import os

from split_dataset import split_data
from tokenize_dataset import tokenize

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dir_name = os.path.dirname(os.path.abspath(__file__))
    tokenized_data_path = os.path.join(dir_name, 'newsqa-data-tokenized-v1.csv')

    tokenize(output_path=tokenized_data_path)
    split_data(dataset_path=tokenized_data_path)
