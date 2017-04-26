# -*- coding: utf-8 -*-
import io
import os
import unittest

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
    @classmethod
    def setUpClass(cls):
        cls.newsqa_dataset = NewsQaDataset()

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
        combined_data_path = os.path.join(dir_name, '..', 'combined-newsqa-data-v1.csv')

        dataset = NewsQaDataset.load_combined(combined_data_path)

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
