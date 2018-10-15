import logging
import os
import csv

import pandas as pd

_dir_name = os.path.dirname(os.path.abspath(__file__))


def _read_csv(file_path):
    res = []
    with open(file_path) as csvfile:
        headers = next(csvfile)
        headers = headers.strip().split(',')
        story_id_index = headers.index('story_id')
        story_text_index = headers.index('story_text')
        question_index = headers.index('question')
        answer_index = headers.index('answer_token_ranges')
        _reader = csv.reader(csvfile)
        for i, row in enumerate(_reader):
            res.append({"story_id": row[story_id_index],
                        "story_text": row[story_text_index],
                        "question": row[question_index],
                        "answer_token_ranges": row[answer_index]})
    return res


def _write_to_csv(data, path):
    logging.info("Writing %d rows to %s", len(data), path)
    pd.DataFrame(data=data).to_csv(path,
                                   columns=["story_id", "story_text", "question", "answer_token_ranges"],
                                   index=False, encoding='utf-8')


def simplify(output_dir_path='split_data'):

    for path in ['train.csv', 'dev.csv', 'test.csv']:
        path = os.path.join(output_dir_path, path)
        data = _read_csv(path)
        _write_to_csv(data, path)
