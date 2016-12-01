# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import io
import json
import logging
import os
import re
import tarfile

import pandas as pd
import six
import tqdm


def strip_empty_strings(strings):
    while strings and strings[-1] == "":
        del strings[-1]
    return strings


class NewsQaDataset(object):
    def __init__(self, cnn_stories_path=None, dataset_path=None, log_level=logging.INFO):
        if not six.PY2:
            raise Exception("Sorry, the loading logic only works with Python 2 for now.")

        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            self._logger.setLevel(log_level)
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            formatter = logging.Formatter(
                '[%(levelname)s] %(asctime)s - %(filename)s::%(funcName)s\n%(message)s')
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

        dirname = os.path.dirname(os.path.abspath(__file__))
        if cnn_stories_path is None:
            cnn_stories_path = os.path.join(dirname, 'cnn_stories.tgz')
        if not os.path.exists(cnn_stories_path):
            raise Exception(
                "`%s` was not found.\nFor legal reasons, you must first download the stories on "
                "your own from http://cs.nyu.edu/~kcho/DMQA/" % cnn_stories_path)
        # TODO Handle dataset in a zipped file.
        if dataset_path is None:
            dataset_path = os.path.join(dirname, 'newsqa-data-v1.csv')
        if not os.path.exists(dataset_path):
            raise Exception(
                "`%s` was not found.\nFor legal reasons, you must first accept the terms "
                "and download the dataset from "
                "https://datasets.maluuba.com/NewsQA/dl" % dataset_path)

        self._logger.info("Loading dataset from `%s`...", dataset_path)
        self.dataset = pd.read_csv(dataset_path,
                                   encoding='utf-8',
                                   na_values=dict(question=[]),
                                   keep_default_na=False)
        remaining_story_ids = set(self.dataset['story_id'])
        self._logger.info("Loading stories from `%s`...", cnn_stories_path)

        with io.open(os.path.join(dirname, 'stories_requiring_extra_newline.csv'),
                     'r', encoding='utf8') as f:
            stories_requiring_extra_newline = set(f.read().split('\n'))

        with io.open(os.path.join(dirname, 'stories_requiring_two_extra_newlines.csv'),
                     'r', encoding='utf8') as f:
            stories_requiring_two_extra_newlines = set(f.read().split('\n'))

        with io.open(os.path.join(dirname, 'stories_to_decode_specially.csv'),
                     'r', encoding='utf8') as f:
            stories_to_decode_specially = set(f.read().split('\n'))

        story_id_to_text = {}
        with tarfile.open(cnn_stories_path, mode='r:gz', encoding='utf-8') as t:
            highlight_indicator = '@highlight'

            copyright_line_pattern = re.compile(
                "^(Copyright|Entire contents of this article copyright, )")
            with tqdm.tqdm(total=len(remaining_story_ids),
                           mininterval=2, unit_scale=True, unit=" stories",
                           desc="Getting story texts") as pbar:
                for member in t.getmembers():
                    story_id = member.name
                    if story_id in remaining_story_ids:
                        remaining_story_ids.remove(story_id)
                        story_file = t.extractfile(member)

                        # Correct discrepancies in stories.
                        # Problems are caused by using several programming languages and libraries.
                        # When ingesting the stories, we started with Python 2.
                        # After dealing with unicode issues, we tried switching to Python 3.
                        # That caused inconsistency problems so we switched back to Python 2.
                        # Furthermore, when crowdsourcing, JavaScript and HTML templating perturbed
                        # the stories.
                        # So here we map the text to be compatible with the indices.
                        if story_id in stories_to_decode_specially:
                            lines = map(lambda s: u"".join(six.unichr(ord(c)) for c in s.strip()),
                                        story_file.readlines())
                        else:
                            lines = map(lambda s: s.strip().decode('utf-8'),
                                        story_file.readlines())

                        story_file.close()
                        if not six.PY2:
                            lines = list(lines)
                        highlights_start = lines.index(highlight_indicator)
                        story_lines = lines[:highlights_start]
                        story_lines = strip_empty_strings(story_lines)
                        while len(story_lines) > 1 and copyright_line_pattern.search(
                                story_lines[-1]):
                            story_lines = strip_empty_strings(story_lines[:-2])
                        if story_id in stories_requiring_two_extra_newlines:
                            story_text = '\r\r\n'.join(story_lines)
                        elif story_id in stories_requiring_extra_newline:
                            story_text = '\r\n'.join(story_lines)
                        else:
                            story_text = '\n'.join(story_lines)

                        story_text = story_text.replace(u'\xe2\x80\xa2', u'\xe2\u20ac\xa2')
                        story_text = story_text.replace(u'\xe2\x82\xac', u'\xe2\u201a\xac')
                        if story_id in stories_to_decode_specially:
                            story_text = story_text.replace(u'\xe9', u'\xc3\xa9')
                        story_id_to_text[story_id] = story_text

                        pbar.update()

                        if len(remaining_story_ids) == 0:
                            break

        for index, row in tqdm.tqdm(self.dataset.iterrows(),
                                    total=len(self.dataset),
                                    mininterval=2, unit_scale=True, unit=" questions",
                                    desc="Setting story texts"):
            # Set story_text since we cannot include it in the dataset.
            story_text = story_id_to_text[row['story_id']]
            self.dataset.set_value(index, 'story_text', story_text)

            # Handle endings that are too large.
            answer_char_ranges = row['answer_char_ranges'].split('|')
            updated_answer_char_ranges = []
            for user_answer_char_ranges in answer_char_ranges:
                updated_user_answer_char_ranges = []
                for char_range in user_answer_char_ranges.split(','):
                    if char_range != 'None':
                        start, end = map(int, char_range.split(':'))
                        if end > len(story_text):
                            end = len(story_text)
                        updated_user_answer_char_ranges.append('%d:%d' % (start, end))
                    else:
                        updated_user_answer_char_ranges.append(char_range)
                updated_user_answer_char_ranges = ','.join(updated_user_answer_char_ranges)
                updated_answer_char_ranges.append(updated_user_answer_char_ranges)
            updated_answer_char_ranges = '|'.join(updated_answer_char_ranges)
            self.dataset.set_value(index, 'answer_char_ranges', updated_answer_char_ranges)

        # TODO Add tokenized story text and other fields from preprocessing scripts.

        self._logger.info("Done loading dataset.")

    def dump(self, path):
        """
        Export the combined dataset, with stories, to a file.

        :param path: The path to write the dataset to.
        """
        logging.info("Packaging dataset to %s", path)
        self.dataset.to_csv(path, index=False, encoding='utf-8')

    def get_vocab_len(self):
        """
        :return: Approximate vocabulary size.
        """
        vocab = set()
        for _, row in tqdm.tqdm(self.dataset.iterrows(),
                                total=len(self.dataset),
                                mininterval=2, unit_scale=True, unit=" questions",
                                desc="Gathering vocab"):
            vocab.update(row['story_text'].lower().split())
        print("Vocabulary length: %s" % len(vocab))
        return len(vocab)

    def get_answers(self, include_no_answers=False):
        answers = []
        for index, row in self.dataset.iterrows():

            # If there are no validated answers, use the ones that are provided.
            if pd.isnull(row['validated_answers']):
                # Ignore per selection splits.
                char_ranges = row['answer_char_ranges'].replace('|', ',').split(',')

            else:
                # Prefer validated answers.
                validated_answers_dict = json.loads(row['validated_answers'])
                char_ranges = []
                for k, v in validated_answers_dict.items():
                    char_ranges += v * [k]

            for char_range in char_ranges:
                if include_no_answers and char_range.lower() == "none":
                    answers.append(None)
                elif ':' in char_range:
                    left, right = char_range.split(':')
                    left = int(left)
                    right = int(right)
                    answer = row['story_text'][left:right]
                    answers.append(answer)

        return pd.Series(answers)

    def get_answer_lengths_words(self, max_length=-1):

        def get_word_count(answer):
            return len(answer.split())

        lengths = self.get_answers().apply(get_word_count)
        if max_length >= 0:
            lengths = lengths[lengths <= max_length]
        return lengths

    def get_questions_and_answers(self, include_no_answers=False):

        qa_map = {}

        def add_q_a_pair_to_map(q, a):

            q = q.lower().strip().strip('.?!').strip()
            if not q:
                q = "no_question"

            if q in qa_map:
                qa_map[q].append(a)
            else:
                qa_map[q] = [a]

        for _, row in tqdm.tqdm(self.dataset.iterrows(),
                                total=len(self.dataset),
                                mininterval=2, unit_scale=True, unit='questions',
                                desc="Gathering answers"):

            # If there's no validated answers, use the ones that are provided
            if pd.isnull(row['validated_answers']):
                char_ranges = row['answer_char_ranges'].replace('|', ',').split(',')

            # But prefer validated answers
            else:
                validated_answers_dict = json.loads(row['validated_answers'])
                char_ranges = []
                for k, v in validated_answers_dict.items():
                    char_ranges += v * [k]

            for char_range in char_ranges:
                if include_no_answers and char_range.lower() == "none":
                    add_q_a_pair_to_map(row['question'], "")

                elif ':' in char_range:
                    left, right = char_range.split(':')
                    left = int(left)
                    right = int(right)
                    answer = row['story_text'][left:right]
                    add_q_a_pair_to_map(row['question'], answer)

        return pd.DataFrame(data=list(qa_map.items()), columns=['question', 'answers'])

    def get_questions(self):
        return pd.Series(self.dataset['question'].dropna())

    def save_dataset_as_json_by_columns(self, path, n_entries=None):
        if not n_entries or n_entries > len(self.dataset):
            df = self.dataset
        else:
            df = self.dataset.head(n_entries)

        res = df.to_json(path, force_ascii=False)

        return res

    def save_dataset_as_json_by_rows(self, path, n_entries=None):
        if not n_entries or n_entries > len(self.dataset):
            df = self.dataset
        else:
            df = self.dataset.head(n_entries)

        data_dict = {}
        for index, row in df.iterrows():
            data_dict[str(index)] = dict(row)

        with codecs.open(path, 'w', encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False)
