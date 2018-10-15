import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

try:
    # Prefer a more specific path.
    from maluuba.newsqa.data_processing import NewsQaDataset
except:
    from data_processing import NewsQaDataset

_dir_name = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger('newsqa')


def split_data(dataset_path, output_dir_path='split_data'):
    original = NewsQaDataset.load_combined(dataset_path)

    logger.info("Loading story ID's split.")
    train_story_ids = set(
        pd.read_csv(os.path.join(_dir_name, 'train_story_ids.csv'))['story_id'].values)
    dev_story_ids = set(
        pd.read_csv(os.path.join(_dir_name, 'dev_story_ids.csv'))['story_id'].values)
    test_story_ids = set(
        pd.read_csv(os.path.join(_dir_name, 'test_story_ids.csv'))['story_id'].values)

    train_data = []
    dev_data = []
    test_data = []

    for row in tqdm(original.itertuples(), total=len(original),
                    mininterval=2, unit_scale=True, unit=" questions",
                    desc="Splitting data"):
        story_id = row.story_id

        # Filter out when no answer was picked because these weren't used in the original paper.
        # FIXME Soon, if data was tokenized first, then it won't have answer_char_ranges, so we should check something else.
        # See the FIXME in the tokenizer for what field to check.
        answer_char_ranges = row.answer_char_ranges.split('|')
        none_count = answer_char_ranges.count('None')
        if none_count == len(answer_char_ranges):
            continue
        if story_id in train_story_ids:
            train_data.append(row)
        elif story_id in dev_story_ids:
            dev_data.append(row)
        elif story_id in test_story_ids:
            test_data.append(row)
        else:
            logger.warning(
                "%s is not in train, dev, nor test", story_id)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    def _write_to_csv(data, path):
        logger.info("Writing %d rows to %s", len(data), path)
        pd.DataFrame(data=data).to_csv(path,
                                       columns=original.columns.values,
                                       index=False, encoding='utf-8')

    assert len(train_data) == 92549, "Incorrect amount of training data."
    assert len(dev_data) == 5166, "Incorrect amount of validation data."
    assert len(test_data) == 5126, "Incorrect amount of test data."

    logger.info("Writing split data to %s", output_dir_path)
    _write_to_csv(train_data, os.path.join(output_dir_path, 'train.csv'))
    _write_to_csv(dev_data, os.path.join(output_dir_path, 'dev.csv'))
    _write_to_csv(test_data, os.path.join(output_dir_path, 'test.csv'))


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(_dir_name))
    default_dataset_path = os.path.join(project_root, 'combined-newsqa-data-v1.csv')
    default_output_dir = os.path.join(_dir_name, 'split_data')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default=default_dataset_path,
                        help="The path to the dataset to split. Default: %s" % default_dataset_path)
    parser.add_argument('--output_dir_path', '--output_dir', default=default_output_dir,
                        help="The path folder to put the split up data. Default: %s"
                             % default_output_dir)
    args = parser.parse_args()
    split_data(args.dataset_path, args.output_dir_path)
