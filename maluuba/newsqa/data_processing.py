# -*- coding: utf-8 -*-
from __future__ import print_function

import codecs
import io
import json
import logging
import os
import re
import tarfile
from collections import Counter
from operator import itemgetter

import numpy as np
import pandas as pd
import six
import tqdm


def strip_empty_strings(strings):
    while strings and strings[-1] == "":
        del strings[-1]
    return strings


def _get_logger(log_level=logging.INFO):
    result = logging.getLogger('newsqa')
    if not result.handlers:
        # Explicitly only set the log level if the logger hasn't been set up yet.
        result.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(filename)s::%(funcName)s\n%(message)s')
        ch.setFormatter(formatter)
        result.addHandler(ch)
    return result


class NewsQaDataset(object):
    def __init__(self, cnn_stories_path=None, dataset_path=None, log_level=logging.INFO,
                 combined_data_path=None):
        self._logger = _get_logger(log_level)

        if combined_data_path:
            self.dataset = self.load_combined(combined_data_path)
            self.version = self._get_version(combined_data_path)
            return

        if not six.PY2:
            raise Exception("Sorry, the loading logic only works with Python 2 for now.")

        dirname = os.path.dirname(os.path.abspath(__file__))
        if cnn_stories_path is None:
            cnn_stories_path = os.path.join(dirname, 'cnn_stories.tgz')
        if not os.path.exists(cnn_stories_path):
            raise Exception(
                "`%s` was not found.\nFor legal reasons, you must first download the stories on "
                "your own from http://cs.nyu.edu/~kcho/DMQA/" % cnn_stories_path)
        if dataset_path is None:
            dataset_path = os.path.join(dirname, 'newsqa-data-v1.csv')
        if not os.path.exists(dataset_path):
            zipped_dataset_paths = list(filter(os.path.exists,
                [
                    os.path.join(os.path.dirname(dataset_path), 'newsqa-data-v1.tar.gz'),
                    os.path.join(os.path.dirname(dataset_path), 'newsqa.tar.gz'),
                ]))
            if len(zipped_dataset_paths) > 0:
                zipped_dataset_path = zipped_dataset_paths[0]
                self._logger.info("Will use zipped dataset at `%s`.", zipped_dataset_path)
                with tarfile.open(zipped_dataset_path, mode='r:gz', encoding='utf-8') as t:
                    extraction_destination_path = os.path.dirname(dataset_path)
                    self._logger.info("Extracting `%s` to `%s`.", zipped_dataset_path, extraction_destination_path)
                    t.extractall(path=extraction_destination_path)
            else:
                raise Exception(
                    "`%s` was not found.\nFor legal reasons, you must first accept the terms "
                    "and download the dataset from "
                    "https://msropendata.com/datasets/939b1042-6402-4697-9c15-7a28de7e1321"
                    "\n See the README in the root of this repo for more details." % dataset_path)

        self.version = self._get_version(dataset_path)

        self._logger.info("Loading dataset from `%s`...", dataset_path)
        # It's not really combined but it's okay because the method still works
        # to load data with missing columns.
        self.dataset = self.load_combined(dataset_path)

        remaining_story_ids = set(self.dataset['story_id'])
        self._logger.info("Loading stories from `%s`...", cnn_stories_path)

        with io.open(os.path.join(dirname, 'stories_requiring_extra_newline.csv'),
                     'r', encoding='utf-8') as f:
            stories_requiring_extra_newline = set(f.read().split('\n'))

        with io.open(os.path.join(dirname, 'stories_requiring_two_extra_newlines.csv'),
                     'r', encoding='utf-8') as f:
            stories_requiring_two_extra_newlines = set(f.read().split('\n'))

        with io.open(os.path.join(dirname, 'stories_to_decode_specially.csv'),
                     'r', encoding='utf-8') as f:
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
                            story_text = '\n\n\n'.join(story_lines)
                        elif story_id in stories_requiring_extra_newline:
                            story_text = '\n\n'.join(story_lines)
                        else:
                            story_text = '\n'.join(story_lines)

                        story_text = story_text.replace(u'\xe2\x80\xa2', u'\xe2\u20ac\xa2')
                        story_text = story_text.replace(u'\xe2\x82\xac', u'\xe2\u201a\xac')
                        story_text = story_text.replace('\r', '\n')
                        if story_id in stories_to_decode_specially:
                            story_text = story_text.replace(u'\xe9', u'\xc3\xa9')
                        story_id_to_text[story_id] = story_text

                        pbar.update()

                        if len(remaining_story_ids) == 0:
                            break

        for row in tqdm.tqdm(self.dataset.itertuples(),
                             total=len(self.dataset),
                             mininterval=2, unit_scale=True, unit=" questions",
                             desc="Setting story texts"):
            # Set story_text since we cannot include it in the dataset.
            story_text = story_id_to_text[row.story_id]
            self.dataset.at[row.Index, 'story_text'] = story_text

            # Handle endings that are too large.
            answer_char_ranges = row.answer_char_ranges.split('|')
            updated_answer_char_ranges = []
            ranges_updated = False
            for user_answer_char_ranges in answer_char_ranges:
                updated_user_answer_char_ranges = []
                for char_range in user_answer_char_ranges.split(','):
                    if char_range != 'None':
                        start, end = map(int, char_range.split(':'))
                        if end > len(story_text):
                            ranges_updated = True
                            end = len(story_text)
                        if start < end:
                            updated_user_answer_char_ranges.append('%d:%d' % (start, end))
                        else:
                            # It's unclear why but sometimes the end is after the start.
                            # We'll filter these out.
                            ranges_updated = True
                    else:
                        updated_user_answer_char_ranges.append(char_range)
                if updated_user_answer_char_ranges:
                    updated_user_answer_char_ranges = ','.join(updated_user_answer_char_ranges)
                    updated_answer_char_ranges.append(updated_user_answer_char_ranges)
            if ranges_updated:
                updated_answer_char_ranges = '|'.join(updated_answer_char_ranges)
                self.dataset.at[row.Index, 'answer_char_ranges'] = updated_answer_char_ranges

            if row.validated_answers and not pd.isnull(row.validated_answers):
                updated_validated_answers = {}
                validated_answers = json.loads(row.validated_answers)
                for char_range, count in six.iteritems(validated_answers):
                    if ':' in char_range:
                        start, end = map(int, char_range.split(':'))
                        if end > len(story_text):
                            ranges_updated = True
                            end = len(story_text)
                        if start < end:
                            char_range = '{}:{}'.format(start, end)
                            updated_validated_answers[char_range] = count
                        else:
                            # It's unclear why but sometimes the end is after the start.
                            # We'll filter these out.
                            ranges_updated = True
                    else:
                        updated_validated_answers[char_range] = count
                if ranges_updated:
                    updated_validated_answers = json.dumps(updated_validated_answers,
                                                           ensure_ascii=False, separators=(',', ':'))
                    self.dataset.at[row.Index, 'validated_answers'] = updated_validated_answers

        self._logger.info("Done loading dataset.")

    @staticmethod
    def load_combined(path):
        """
        :param path: The path of data to load.
        :return: A `DataFrame` containing the data from `path`.
        :rtype: pandas.DataFrame
        """

        logger = _get_logger()

        logger.info("Loading data from `%s`...", path)

        result = pd.read_csv(path,
                             encoding='utf-8',
                             dtype=dict(is_answer_absent=float),
                             na_values=dict(question=[], story_text=[], validated_answers=[]),
                             keep_default_na=False)

        if 'story_text' in result.keys():
            for row in tqdm.tqdm(result.itertuples(),
                                 total=len(result),
                                 mininterval=2, unit_scale=True, unit=" questions",
                                 desc="Adjusting story texts"):
                # Correct story_text to make indices work right.
                story_text = row.story_text.replace('\r\n', '\n')
                result.at[row.Index, 'story_text'] = story_text

        return result

    def _get_version(self, path):
        m = re.match(r'^.*-v(([\d.])*\d+).[^.]*$', path)
        if not m:
            raise ValueError("Version number not found in `{}`.".format(path))
        return m.group(1)

    def _map_answers(self, answers):
        result = []
        for a in answers.split('|'):
            user_answers = []
            result.append(dict(sourcerAnswers=user_answers))
            for r in a.split(','):
                if r == 'None':
                    user_answers.append(dict(noAnswer=True))
                else:
                    s, e = map(int, r.split(':'))
                    user_answers.append(dict(s=s, e=e))
        return result

    def export_shareable(self, path, package_path=None):
        """
        Export the dataset without the stories so that it can be shared.

        :param path: The path to write the dataset to.
        :param package_path: (Optional) If given, the path to write the tar.gz for the website.
        """
        self._logger.info("Exporting dataset to %s", path)
        columns = list(self.dataset.columns.values)
        columns_to_remove = [
            'story_title',
            'story_text',
            'popular_answer_char_ranges',
            'popular_answers (for humans to read)',
        ]
        for col in columns_to_remove:
            try:
                columns.remove(col)
            except:
                pass
        self.dataset.to_csv(path, columns=columns, index=False, encoding='utf-8')

        if package_path:
            dirname = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(dirname))
            os.chdir(os.path.dirname(package_path))
            with tarfile.open(os.path.basename(package_path), 'w|gz', encoding='utf-8') as t:
                t.add(os.path.join(project_root, 'README-distribution.md'), arcname='README.md')
                t.add(os.path.join(project_root, 'LICENSE.txt'), arcname='LICENSE.txt')
                t.add(path, arcname=os.path.basename(path))

    def dump(self, path):
        """
        Export the combined dataset, with stories, to a file.

        :param path: The path to write the dataset to.
        """
        self._logger.info("Packaging dataset to `%s`.", path)
        if path.endswith('.json'):
            data = self.to_dict()
            # Most reliable way to write UTF-8 JSON as described: https://stackoverflow.com/a/18337754/1226799
            data = json.dumps(data, ensure_ascii=False, separators=(',', ':'), encoding='utf-8')
            with io.open(path, 'w', encoding='utf-8') as f:
                f.write(unicode(data))
        else:
            if not path.endswith('.csv'):
                self._logger.warning("Writing data as CSV to `%s`.", path)
            # Default for backwards compatibility.
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
        for row in tqdm.tqdm(self.dataset.itertuples(),
                             total=len(self.dataset),
                             mininterval=2, unit_scale=True, unit=" questions",
                             desc="Gathering answers"):

            # Prefer validated answers.
            # If there are no validated answers, use the ones that are provided.
            if not row.validated_answers or pd.isnull(row.validated_answers):
                # Ignore per selection splits.
                char_ranges = row.answer_char_ranges.replace('|', ',').split(',')
            else:
                validated_answers_dict = json.loads(row.validated_answers)
                char_ranges = []
                for k, v in validated_answers_dict.items():
                    char_ranges += v * [k]

            for char_range in char_ranges:
                if include_no_answers and char_range.lower() == 'none':
                    answers.append(None)
                elif ':' in char_range:
                    start, end = map(int, char_range.split(':'))
                    answer = row.story_text[start:end]
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

        for row in tqdm.tqdm(self.dataset.itertuples(),
                             total=len(self.dataset),
                             mininterval=2, unit_scale=True, unit=" questions",
                             desc="Gathering answers"):

            # Prefer validated answers.
            # If there are no validated answers, use the ones that are provided.
            if not row.validated_answers or pd.isnull(row.validated_answers):
                char_ranges = row.answer_char_ranges.replace('|', ',').split(',')
            else:
                validated_answers_dict = json.loads(row.validated_answers)
                char_ranges = []
                for k, v in validated_answers_dict.items():
                    char_ranges += v * [k]

            for char_range in char_ranges:
                if include_no_answers and char_range.lower() == 'none':
                    add_q_a_pair_to_map(row.question, "")

                elif ':' in char_range:
                    start, end = map(int, char_range.split(':'))
                    answer = row.story_text[start:end]
                    add_q_a_pair_to_map(row.question, answer)

        return pd.DataFrame(data=list(qa_map.items()), columns=['question', 'answers'])

    def get_average_answer_length_over_questions(self):

        def get_word_count(answer):
            return len(answer.split())

        qas = self.get_questions_and_answers()
        avg_ans_lengths = np.zeros(len(qas))
        for index, row in qas.iterrows():
            avg_ans_lengths[index] = np.average([get_word_count(answer)
                                                 for answer in row['answers']])

        return pd.Series(avg_ans_lengths)

    def get_consensus_answer(self, row):
        """
        Gets the consensus answer.

        Note that there cannot be multiple since we only ran validation when there was no consensus.
        Then each validator was only allowed to pick one option and we used an odd number of validators.

        :param row: A row in the dataset.
        :return: The answer with majority consensus.
            Can be `(None, None)` if it was agreed that there was no answer or it was a bad question.
        :rtype: tuple
        """
        answer_char_start, answer_char_end = None, None
        if row.validated_answers:
            validated_answers = json.loads(row.validated_answers)
            answer, max_count = max(six.iteritems(validated_answers), key=itemgetter(1))
            total_count = sum(six.itervalues(validated_answers))
            if max_count >= total_count / 2.0:
                if answer != 'none' and answer != 'bad_question':
                    answer_char_start, answer_char_end = map(int, answer.split(':'))
                else:
                    # No valid answer.
                    pass
        else:
            # Check row.answer_char_ranges for most common answer.
            # No validation was done so there must be an answer with consensus.
            answers = Counter()
            for user_answer in row.answer_char_ranges.split('|'):
                for ans in user_answer.split(','):
                    answers[ans] += 1
            top_answer = answers.most_common(1)
            if top_answer:
                top_answer, count = top_answer[0]
                if ':' in top_answer:
                    answer_char_start, answer_char_end = map(int, top_answer.split(':'))

        return answer_char_start, answer_char_end

    def get_question_types(self, num_most_common=6):
        # Note: Would be nice not to make a series and just keep track of the counts
        # but we couldn't get it to plot nicely in a bar plot.
        result = Counter()
        for _, row in tqdm.tqdm(self.dataset.iterrows(),
                                total=len(self.dataset),
                                mininterval=2, unit_scale=True, unit='questions',
                                desc="Categorizing questions"):
            q = row['question']
            split = q.split()
            # Ignore empty questions, they shouldn't happen.
            if split:
                # Use the first token as the question type.
                question_type = split[0].lower()
                result[question_type] += 1

        result = result.most_common(num_most_common)
        num_remaining = len(self.dataset) - sum(map(itemgetter(1), result))
        result.append(('*other', num_remaining))
        result = sorted(result, key=itemgetter(1), reverse=True)

        result = pd.DataFrame(dict(question_type=list(map(itemgetter(0), result)),
                                   count=list(map(itemgetter(1), result))))

        return result

    def get_story_lengths_words(self):

        def get_word_count(story):
            return len(story.split())

        de_duped = self.dataset.drop_duplicates(subset='story_id')
        return de_duped['story_text'].apply(get_word_count)

    def get_questions(self):
        return pd.Series(self.dataset['question'].dropna())

    def get_question_lengths_words(self, max_length=-1):

        def get_word_count(question):
            return len(question.split())

        lengths = self.get_questions().apply(get_word_count)
        if max_length >= 0:
            lengths = lengths[lengths <= max_length]
        return lengths

    def get_questions_without_answers(self):
        questions_without_answers = []
        for index, row in self.dataset.iterrows():
            if not pd.isnull(row['question']) \
                    and not (pd.isnull(row['answer_char_ranges'])
                             and pd.isnull(row['validated_answers'])) \
                    and ':' not in row['answer_char_ranges']:
                questions_without_answers.append(row['question'])

        return questions_without_answers

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

    def get_all_qas_for_story_ids(self, story_ids=None, n_stories=-1, include_no_answers=False):

        data = {}
        for story_id, matching_df in self.dataset.groupby('story_id'):

            if story_ids and not story_id in story_ids:
                continue
            if n_stories >= 0 and len(data.keys()) >= n_stories:
                break

            entry = {}
            for _, row in matching_df.iterrows():

                # Initialization case
                if 'story_text' not in entry:
                    # Note: there are no titles in the dataset.
                    entry['story_title'] = row['story_title']
                    entry['story_text'] = row['story_text']
                    entry['qa_pairs'] = []

                # Prefer validated answers; if none fallback to regular ones
                answers = []
                if not pd.isnull(row['validated_answers']):
                    validated_answers_dict = json.loads(row['validated_answers'])
                    answers += validated_answers_dict.keys()

                else:
                    answers_split = row['answer_char_ranges'].split("|")
                    for answer in answers_split:
                        if answer not in answers:
                            answers.append(answer)

                if not include_no_answers:
                    answers = [a for a in answers if a.lower() != "none"]

                entry['qa_pairs'].append(
                    {'question': row['question'], 'answers': "|".join(answers)})

            data[story_id] = entry

        return data

    def to_dict(self):
        """
        :return: The data in a `dict`.
        :rtype: dict
        """
        data = []
        cache = dict()

        dir_name = os.path.dirname(os.path.abspath(__file__))

        train_story_ids = set(
            pd.read_csv(os.path.join(dir_name, 'train_story_ids.csv'))['story_id'].values)
        dev_story_ids = set(
            pd.read_csv(os.path.join(dir_name, 'dev_story_ids.csv'))['story_id'].values)
        test_story_ids = set(
            pd.read_csv(os.path.join(dir_name, 'test_story_ids.csv'))['story_id'].values)

        def _get_data_type(story_id):
            if story_id in train_story_ids:
                return 'train'
            elif story_id in dev_story_ids:
                return 'dev'
            elif story_id in test_story_ids:
                return 'test'
            else:
                return ValueError("{} not found in any story ID set.".format(story_id))

        for row in tqdm.tqdm(self.dataset.itertuples(),
                             total=len(self.dataset),
                             mininterval=2, unit_scale=True, unit=" questions",
                             desc="Building json"):
            questions = cache.get(row.story_id)
            if questions is None:
                questions = []
                datum = dict(storyId=row.story_id,
                             type=_get_data_type(row.story_id),
                             text=row.story_text,
                             questions=questions)
                cache[row.story_id] = questions
                data.append(datum)
            q = dict(
                q=row.question,
                answers=self._map_answers(row.answer_char_ranges),
                isAnswerAbsent=row.is_answer_absent,
            )
            if row.is_question_bad != '?':
                q['isQuestionBad'] = float(row.is_question_bad)
            if row.validated_answers and not pd.isnull(row.validated_answers):
                validated_answers = json.loads(row.validated_answers)
                q['validatedAnswers'] = []
                for answer, count in six.iteritems(validated_answers):
                    answer_item = dict(count=count)
                    if answer == 'none':
                        answer_item['noAnswer'] = True
                    elif answer == 'bad_question':
                        answer_item['badQuestion'] = True
                    else:
                        s, e = map(int, answer.split(':'))
                        answer_item['s'] = s
                        answer_item['e'] = e
                    q['validatedAnswers'].append(answer_item)
            consensus_start, consensus_end = self.get_consensus_answer(row)
            if consensus_start is None and consensus_end is None:
                if q.get('isQuestionBad', 0) >= 0.5:
                    q['consensus'] = dict(badQuestion=True)
                else:
                    q['consensus'] = dict(noAnswer=True)
            else:
                q['consensus'] = dict(s=consensus_start, e=consensus_end)
            questions.append(q)

        data = dict(data=data, version=self.version)
        return data
