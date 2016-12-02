import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

dir_name = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(dir_name))
default_dataset_path = os.path.join(project_root, 'combined-newsqa-data-v1.csv')
default_output_dir = os.path.join(dir_name, 'split_data')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default=default_dataset_path,
                    help="The path to the dataset to split. Default: %s" % default_dataset_path)
parser.add_argument('--output_dir', default=default_output_dir,
                    help="The path folder to put the split up data. Default: %s"
                         % default_output_dir)
args = parser.parse_args()

original = pd.read_csv(args.dataset_path,
                       encoding='utf-8',
                       na_values=dict(question=[]),
                       keep_default_na=False)

train_story_ids = set(pd.read_csv(os.path.join(dir_name, 'train_story_ids.csv'))['story_id'].values)
dev_story_ids = set(pd.read_csv(os.path.join(dir_name, 'dev_story_ids.csv'))['story_id'].values)
test_story_ids = set(pd.read_csv(os.path.join(dir_name, 'test_story_ids.csv'))['story_id'].values)

train_data = []
dev_data = []
test_data = []

for _, row in tqdm(original.iterrows(), total=len(original),
                   mininterval=2, unit_scale=True, unit=" questions",
                   desc="Splitting data"):
    story_id = row['story_id']
    if story_id in train_story_ids:
        train_data.append(row)
    elif story_id in dev_story_ids:
        dev_data.append(row)
    elif story_id in test_story_ids:
        test_data.append(row)
    else:
        logging.warning(
            "%s is not in train, dev, nor test", story_id)

train_out = pd.DataFrame(data=train_data)
dev_out = pd.DataFrame(data=dev_data)
tests_out = pd.DataFrame(data=test_data)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logging.info("Writing split data to %s", args.output_dir)
train_out.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False, encoding='utf-8')
dev_out.to_csv(os.path.join(args.output_dir, 'dev.csv'), index=False, encoding='utf-8')
tests_out.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False, encoding='utf-8')
