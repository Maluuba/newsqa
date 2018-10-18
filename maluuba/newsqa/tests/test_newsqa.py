# -*- coding: utf-8 -*-
import io
import json
import os
import unittest
from collections import namedtuple

from tqdm import tqdm

from maluuba.newsqa.data_processing import NewsQaDataset

_TestRow = namedtuple('TestRow', ['story_id', 'question', 'answer_char_ranges', 'is_answer_absent',
                                  'is_question_bad', 'validated_answers', 'story_text'])


def _get_answers(row):
    result = set()
    for user_spans in row.answer_char_ranges.split('|'):
        for span in user_spans.split(','):
            if span == 'None':
                continue
            s, e = map(int, span.split(':'))
            result.add(row.story_text[s:e])
    return result


def _is_token_ending(char):
    return char.isspace() or char in TestNewsQa._sentence_endings


class TestNewsQa(unittest.TestCase):
    _sentence_endings = set(u'.!?"”)')

    @classmethod
    def setUpClass(cls):
        cls.newsqa_dataset = NewsQaDataset()

    def check_corruption(self, dataset):
        corrupt_count = 0
        for row in tqdm(dataset.itertuples(index=False),
                        total=len(dataset),
                        mininterval=2, unit_scale=True, unit=" questions",
                        desc="Checking for possible corruption"):
            story_text = row.story_text
            answer_char_ranges = row.answer_char_ranges.split('|')
            corrupt = False
            for user_answer_char_ranges in answer_char_ranges:
                for char_range in user_answer_char_ranges.split(','):
                    if char_range != 'None':
                        start, end = map(int, char_range.split(':'))
                        if start > len(story_text) or end > len(story_text) \
                                or (start > 0 and not story_text[start - 1].isspace()) \
                                or not _is_token_ending(story_text[end - 1]):
                            corrupt = True
                            corrupt_count += 1
                            break
                if corrupt:
                    break
        corrupt_percent = corrupt_count * 1.0 / len(dataset)
        # Some issues are permitted due to certain characteristics of the original text
        # that aren't worth checking.
        self.assertLess(corrupt_percent, 0.00065,
                        msg="Possibly corrupt: %d/%d (%.2f)%%" % (corrupt_count, len(dataset), corrupt_percent * 100))

    def test_check_corruption(self):
        self.check_corruption(self.newsqa_dataset.dataset)

    def test_dump_json(self):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        combined_data_path = os.path.join(dir_name, '../../../combined-newsqa-data-v1.json')
        combined_data_path = os.path.abspath(combined_data_path)
        self.newsqa_dataset.dump(path=combined_data_path)

        with io.open(combined_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertIn('data', data)
        self.assertEqual(data.get('version'), '1')
        data = data['data']

        self.assertGreater(len(data), 0)
        self.assertEqual(len(data), 12744)

        corrupt_count = 0
        validation_corrupt_count = 0
        num_questions = 0
        for story_item in tqdm(data,
                               mininterval=2, unit_scale=True, unit=" stories",
                               desc="Checking for possible corruption"):
            story_text = story_item['text']
            for question_item in story_item['questions']:
                num_questions += 1
                corrupt = False
                for sourcer_answer_char_ranges in question_item['answers']:
                    for char_range in sourcer_answer_char_ranges['sourcerAnswers']:
                        if not char_range.get('noAnswer'):
                            start, end = char_range['s'], char_range['e']
                            if start > len(story_text) or end > len(story_text) \
                                    or (start > 0 and not story_text[start - 1].isspace()) \
                                    or not _is_token_ending(story_text[end - 1]):
                                corrupt = True
                                corrupt_count += 1
                                break
                    if corrupt:
                        break
                for val in question_item.get('validatedAnswers', ()):
                    start, end = val.get('s'), val.get('e')
                    if start is not None and end is not None \
                            and (start > len(story_text) or end > len(story_text) \
                                 or (start > 0 and not story_text[start - 1].isspace()) \
                                 or not _is_token_ending(story_text[end - 1])):
                        validation_corrupt_count += 1
                        break

        corrupt_percent = corrupt_count * 1.0 / num_questions
        validation_corrupt_percent = validation_corrupt_count * 1.0 / num_questions
        # Some issues are permitted due to certain characteristics of the original text
        # that aren't worth checking.
        self.assertLess(corrupt_percent, 0.00065,
                        msg="Possibly corrupt: %d/%d (%.2f)%%" % (corrupt_count, num_questions, corrupt_percent * 100))
        self.assertLess(validation_corrupt_percent, 0.00065,
                        msg="Possibly corrupt validation: %d/%d (%.2f)%%" % (validation_corrupt_count, num_questions,
                                                                             validation_corrupt_percent * 100))

    def test_entry_0(self):
        """
        Sanity test to make sure the first entry loads.
        """
        row = self.newsqa_dataset.dataset.iloc[0]
        self.assertEqual('./cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story',
                         row['story_id'])
        self.assertEqual("What was the amount of children murdered?", row['question'])
        self.assertEqual('294:297|None|None', row['answer_char_ranges'])
        self.assertEqual(0.0, row['is_answer_absent'])
        self.assertEqual('0.0', row['is_question_bad'])
        self.assertEqual('{"none": 1, "294:297": 2}', row['validated_answers'])
        self.assertEqual("NEW DELHI, India (CNN) -- A high court in nort", row['story_text'][:46])
        self.assertEqual({"19 "}, _get_answers(row))

    def test_entry_6(self):
        """
        Test an entry with commas in the character ranges to show that a user can pick
        multiple answers.
        """
        row = self.newsqa_dataset.dataset.iloc[6]
        self.assertEqual('./cnn/stories/d312173b8c95cc6c206a32cc0acd8a92c1e272d5.story',
                         row['story_id'])
        self.assertEqual("Who is hiring?", row['question'])
        self.assertEqual('334:345|292:297,372:379|4045:4079|301:324', row['answer_char_ranges'])
        self.assertEqual(0, row['is_answer_absent'])
        self.assertEqual('0.0', row['is_question_bad'])
        self.assertEqual('{"301:324": 2}', row['validated_answers'])
        self.assertEqual("CNN affiliates report on where job seekers are", row['story_text'][:46])
        self.assertEqual({"the states ",
                          "census ",
                          "The naval facility in China Lake, ",
                          "2010 ",
                          "the federal government "}, _get_answers(row))

    def test_entry_28(self):
        """
        Test an entry with no is_question_bad indication and no validation.
        """
        row = self.newsqa_dataset.dataset.iloc[28]
        self.assertEqual('./cnn/stories/d41dc7fc05273a37f0aceaf4f3e35a187f12653e.story',
                         row['story_id'])
        self.assertEqual("Who is appearing in court?", row['question'])
        self.assertEqual('303:320|303:352|303:320', row['answer_char_ranges'])
        self.assertEqual(0, row['is_answer_absent'])
        self.assertEqual('?', row['is_question_bad'])
        self.assertEqual('', row['validated_answers'])
        self.assertEqual("(CNN) -- A former government contract", row['story_text'][:37])
        self.assertEqual({"Roy Lynn Oakley, ",
                          "Roy Lynn Oakley, 67, of Roane County, Tennessee, "}, _get_answers(row))

    def test_get_answers(self):
        answers = self.newsqa_dataset.get_answers()
        self.assertEqual(list(answers[:4]),
                         ["19 ", "19 ", "Sudanese region of Darfur ", "Seleia, "])

    def test_get_consensus_answer(self):
        # Top valid answer.
        row = _TestRow(story_id='test0', question="Who did it?",
                       answer_char_ranges='0:3', is_answer_absent=False, is_question_bad=False,
                       validated_answers='{"0:3":1}',
                       story_text="You did it.")
        self.assertTupleEqual((0, 3), self.newsqa_dataset.get_consensus_answer(row))

        # Top valid answer = none.
        row = _TestRow(story_id='test0', question="Who did it?",
                       answer_char_ranges='0:3', is_answer_absent=False, is_question_bad=False,
                       validated_answers='{"0:3":1, "none":2}',
                       story_text="You did it.")
        self.assertTupleEqual((None, None), self.newsqa_dataset.get_consensus_answer(row))

        # No good valid answer.
        row = _TestRow(story_id='test0', question="Who did it?",
                       answer_char_ranges='0:3', is_answer_absent=False, is_question_bad=False,
                       validated_answers='{"0:3":1, "4:7":1, "none":1}',
                       story_text="You did it.")
        self.assertTupleEqual((None, None), self.newsqa_dataset.get_consensus_answer(row))

        # Use answer_char_ranges.
        row = _TestRow(story_id='test0', question="Who did it?",
                       answer_char_ranges='0:3|4:7|0:3', is_answer_absent=False, is_question_bad=False,
                       validated_answers='',
                       story_text="You did it.")
        self.assertTupleEqual((0, 3), self.newsqa_dataset.get_consensus_answer(row))

        # Use answer_char_ranges with none
        row = _TestRow(story_id='test0', question="Who did it?",
                       answer_char_ranges='none|4:7|none', is_answer_absent=False, is_question_bad=False,
                       validated_answers='',
                       story_text="You did it.")
        self.assertTupleEqual((None, None), self.newsqa_dataset.get_consensus_answer(row))

    def test_load_combined(self):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        combined_data_path = os.path.join(dir_name, '../../../combined-newsqa-data-v1.csv')
        combined_data_path = os.path.abspath(combined_data_path)

        if not os.path.exists(combined_data_path):
            self.newsqa_dataset.dump(path=combined_data_path)

        dataset = NewsQaDataset.load_combined(combined_data_path)

        for original_row in tqdm(self.newsqa_dataset.dataset.itertuples(),
                                 desc="Comparing stories",
                                 total=len(self.newsqa_dataset.dataset),
                                 unit_scale=True, mininterval=2, unit=" rows"):
            expected = original_row.story_text
            actual = dataset.iloc[original_row.Index].story_text
            self.assertEqual(expected, actual,
                             msg="Story texts at position %d are not equal."
                                 "\nExpected:\"%s\""
                                 "\n     Got:\"%s\"" % (original_row.Index, repr(expected), repr(actual)))

        row = dataset.iloc[0]
        self.assertEqual('./cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story',
                         row.story_id)
        self.assertEqual("What was the amount of children murdered?", row.question)
        self.assertEqual('294:297|None|None', row['answer_char_ranges'])
        self.assertEqual(0.0, row['is_answer_absent'])
        self.assertEqual('0.0', row['is_question_bad'])
        self.assertEqual('{"none": 1, "294:297": 2}', row['validated_answers'])
        self.assertEqual("NEW DELHI, India (CNN) -- A high court in nort", row.story_text[:46])
        self.assertEqual({"19 "}, _get_answers(row))


if __name__ == '__main__':
    unittest.main()
