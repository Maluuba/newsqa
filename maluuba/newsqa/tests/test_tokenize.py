# -*- coding: utf-8 -*-
from __future__ import print_function

import logging
import os
import unittest

from maluuba.newsqa.data_processing import NewsQaDataset
from maluuba.newsqa.tokenize_dataset import tokenize


def _get_answers(row):
    result = []
    story_tokens = row.story_text.split()
    for span in row.answer_token_ranges.split(','):
        s, e = map(int, span.split(':'))
        result.append(" ".join(story_tokens[s:e]))
    return result


class TestNewsQaTokenize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        tokenized_data_path = os.path.join(dir_name, '..', 'newsqa-data-tokenized-v1.csv')
        tokenized_data_path = os.path.abspath(tokenized_data_path)
        if not os.path.exists(tokenized_data_path):
            combined_data_path = os.path.join(dir_name, '../../../combined-newsqa-data-v1.csv')
            combined_data_path = os.path.abspath(combined_data_path)

            if not os.path.exists(combined_data_path):
                NewsQaDataset().dump(path=combined_data_path)

            tokenize(combined_data_path=combined_data_path, output_path=tokenized_data_path)

        cls.dataset = NewsQaDataset.load_combined(tokenized_data_path)

    def test_entry_0(self):
        """
        Sanity test to make sure the first entry loads.
        """
        row = self.dataset.iloc[0]
        self.assertEqual('./cnn/stories/42d01e187213e86f5fe617fe32e716ff7fa3afc4.story',
                         row.story_id)
        self.assertEqual("What was the amount of children murdered ?", row.question)
        self.assertEqual(0.0, row.is_answer_absent)
        self.assertEqual('0.0', row.is_question_bad)
        self.assertEqual('41,55,82,100,126,138,165,181,204,219,237', row.sentence_starts)
        self.assertEqual('60:61', row.answer_token_ranges)
        self.assertEqual("NEW DELHI , India -LRB- CNN -RRB- -- A high co", row.story_text[:46])
        self.assertEqual(["19"], _get_answers(row))

    def test_entry_6(self):
        """
        Test an entry with commas in the character ranges to show that a user can pick
        multiple answers.
        """
        row = self.dataset.iloc[6]
        self.assertEqual('./cnn/stories/d312173b8c95cc6c206a32cc0acd8a92c1e272d5.story',
                         row.story_id)
        self.assertEqual("Who is hiring ?", row.question)
        self.assertEqual('334:345|292:297,372:379|4045:4079|301:324', row.answer_char_ranges)
        self.assertEqual(0, row.is_answer_absent)
        self.assertEqual('0.0', row.is_question_bad)
        self.assertEqual(
            '25,43,71,94,109,120,131,158,168,188,208,222,260,288,327,332,348,369,378,409,417,428,478,504,541,606,623,636,662,682,689,695,712,732,752,778,808,818,852,883,898,927,956,975,999,1020,1051,1069,1080,1092',
            row.sentence_starts)
        self.assertEqual('56:59', row.answer_token_ranges)
        self.assertEqual("CNN affiliates report on where job seekers are", row.story_text[:46])
        self.assertEqual(["the federal government"], _get_answers(row))

    def test_entry_28(self):
        """
        Test an entry with no is_question_bad indication and no validation.
        """
        row = self.dataset.iloc[28]
        self.assertEqual('./cnn/stories/d41dc7fc05273a37f0aceaf4f3e35a187f12653e.story',
                         row.story_id)
        self.assertEqual("Who is appearing in court ?", row.question)
        self.assertEqual(0, row.is_answer_absent)
        self.assertEqual('?', row.is_question_bad)
        self.assertEqual(
            '34,48,69,99,115,143,217,228,246,265,290,319,338,376,395,425,446,516,542,575,605,640,674,704,751,780,820,847,863',
            row.sentence_starts)
        self.assertEqual('48:51', row.answer_token_ranges)
        self.assertEqual("-LRB- CNN -RRB- -- A former governmen", row.story_text[:37])
        self.assertEqual(["Roy Lynn Oakley"], _get_answers(row))

    def test_answers(self):
        """
        Test a few entries to make sure answers are okay.
        """
        row = self.dataset.iloc[87]
        self.assertEqual("When was the trial due to start ?", row.question)
        self.assertEqual(["Wednesday"], _get_answers(row))

        row = self.dataset.iloc[31221]
        self.assertEqual("Whose rights have not improved under the Taliban ?", row.question)
        self.assertEqual(["Conditions for women"], _get_answers(row))

        row = self.dataset.iloc[45648]
        self.assertEqual("What does Vertu make ?", row.question)
        self.assertEqual(["phones starting at $ 6,000"], _get_answers(row))


if __name__ == '__main__':
    unittest.main()
