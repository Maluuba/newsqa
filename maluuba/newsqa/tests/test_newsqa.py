# -*- coding: utf-8 -*-
import io
import logging
import os
import unittest

from tqdm import tqdm

from maluuba.newsqa.data_processing import NewsQaDataset


def _get_answers(row):
    result = set()
    for user_spans in row.answer_char_ranges.split('|'):
        for span in user_spans.split(','):
            if span == 'None':
                continue
            s, e = map(int, span.split(':'))
            result.add(row.story_text[s:e])
    return result


class TestNewsQa(unittest.TestCase):
    _sentence_endings = set(u'.!?"”)')

    @classmethod
    def setUpClass(cls):
        cls.newsqa_dataset = NewsQaDataset()

    def check_corruption(self, dataset):
        def _is_token_ending(char):
            return char.isspace() or char in TestNewsQa._sentence_endings

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


def _write_to_file(path, story_ids):
    if story_ids:
        with io.open(path, 'r', encoding='utf8') as f:
            story_ids.update(f.read().split('\n'))
        story_ids = filter(None, story_ids)
        with io.open(path, 'w', encoding='utf8') as f:
            f.write(u'\n'.join(sorted(story_ids)))


if __name__ == '__main__':
    unittest.main()
