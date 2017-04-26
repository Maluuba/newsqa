from __future__ import unicode_literals

import io
import logging
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import six
from tqdm import tqdm

try:
    # Prefer a more specific path for when you run from the root of this repo
    # or if the root of the repo is in your path.
    from maluuba.newsqa.data_processing import NewsQaDataset
    import maluuba.newsqa.span_utils as span_utils
except:
    # In case you're running this file from this folder.
    from data_processing import NewsQaDataset
    import span_utils

NEARBY_RANGE_THRESHOLD = 3


def format(text):
    return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')


def pack(dataset, writer):
    for row in tqdm(dataset.itertuples(), total=len(dataset),
                    mininterval=2, unit_scale=True, unit=" questions",
                    desc="Packing"):

        writer.write(row.question)
        writer.write('\n')
        refined_valid_spans = span_utils.valid_span_rack_from_string(
            row.validated_answers, row.story_text)
        # Pack tagged texts
        refined_spans = span_utils.refine_answers(
            span_utils.span_rack_from_string(row.answer_char_ranges),
            row.story_text)

        valid_tagged_texts = span_utils.tag_text_from_span_rack(
            refined_valid_spans,
            row.story_text)
        writer.write(u'%s\n' % format(valid_tagged_texts[0]))

        all_tagged_texts = span_utils.tag_text_from_span_rack(
            refined_spans,
            row.story_text)

        writer.write(u'%s\n' % format(all_tagged_texts[0]))


def unpack(dataset, packed, output_path):
    def _read_unpacked():
        n_sents = int(next(packed).strip())
        return [next(packed).strip() for _ in six.moves.xrange(n_sents)]

    data = []
    for row in tqdm(dataset.itertuples(), total=len(dataset),
                    mininterval=2, unit_scale=True, unit=" questions",
                    desc="Unpacking"):
        question = u' '.join(_read_unpacked())

        # Valid starts
        valid_tagged_text_sents = _read_unpacked()
        valid_tagged_text = ' '.join(valid_tagged_text_sents)

        valid_text_sents = [span_utils.remove_tags(s) for s in valid_tagged_text_sents]
        story_text = span_utils.remove_tags(valid_tagged_text)

        valid_span_rack = span_utils.span_rack_from_tag_text(
            [valid_tagged_text], story_text)
        valid_span_rack = span_utils.nearby_range_merge(
            valid_span_rack, threshold=NEARBY_RANGE_THRESHOLD)

        answer_token_ranges = span_utils.span_rack_to_string(valid_span_rack)
        # Valid ends.

        sentences_ids = [0]
        for s in valid_text_sents:
            sentences_ids.append(len(s.split()) + sentences_ids[-1])
        # No need for the 0 as the first element because each article starts with a sentence.
        sentences_ids = sentences_ids[1:]
        sentence_starts = u','.join(map(str, sentences_ids))

        refined_tagged_text_sents = _read_unpacked()

        if len(answer_token_ranges) == 0:
            # Use refined data.
            refined_tagged_text = u" ".join(refined_tagged_text_sents)

            refined_text_sents = [
                span_utils.remove_tags(s) for s in refined_tagged_text_sents]
            story_text_2 = span_utils.remove_tags(refined_tagged_text)

            refined_span_rack = span_utils.span_rack_from_tag_text(
                [refined_tagged_text], story_text_2)
            refined_span_rack = span_utils.nearby_range_merge(
                refined_span_rack, threshold=NEARBY_RANGE_THRESHOLD)
            answer_ranges_2 = span_utils.span_rack_to_string(refined_span_rack)

            sentences_ids = [0]
            for s in refined_text_sents:
                sentences_ids.append(len(s.split()) + sentences_ids[-1])
            sentences_ids = sentences_ids[1:]
            sentence_starts_2 = ','.join(map(str, sentences_ids))

            if len(answer_ranges_2) == 0:
                answer_token_ranges = '-1:-1'
            else:
                answer_token_ranges = answer_ranges_2
                sentence_starts = sentence_starts_2
                story_text = story_text_2

        # Custom columns because we only want to expose certain fields that have indices mapped.
        datum = row._asdict()
        datum['question'] = question
        datum['answer_token_ranges'] = answer_token_ranges
        datum['sentence_starts'] = sentence_starts
        datum['story_text'] = story_text
        # Remove validated answers since they're for character indices which are wrong now.
        del datum['validated_answers']
        del datum['Index']

        # FIXME Keep `answer_char_ranges` for now since splitting needs it.
        # TODO Add another flag that splitting can use to know to remove these so that it doesn't need answer_char_ranges.

        data.append(datum)

    logging.info("Writing to `%s`.", output_path)
    pd.DataFrame(data=data).to_csv(output_path,
                                   index=False, encoding='utf-8')


def tokenize(cnn_stories='cnn_stories.tgz', csv_dataset='newsqa-data-v1.csv',
             combined_data_path='combined-newsqa-data-v1.csv',
             output_path='newsqa-data-tokenized-v1.csv'):
    newsqa_data = NewsQaDataset(cnn_stories, csv_dataset,
                                combined_data_path=combined_data_path)
    dataset = newsqa_data.dataset

    dir_name = os.path.dirname(os.path.abspath(__file__))
    requirements = (dir_name,
                    os.path.join(dir_name, 'stanford-postagger.jar'),
                    os.path.join(dir_name, 'slf4j-api.jar'))
    for req in requirements:
        if not os.path.exists(req):
            raise Exception("Missing `%s`\n"
                            "Please refer to the README in the root of the project regarding the JAR's required." % req)

    packed_filename = os.path.join(dir_name, csv_dataset + '.pck')
    unpacked_filename = os.path.join(dir_name, csv_dataset + '.tpck')

    logging.info("(1/3) - Packing data to `%s`.", packed_filename)
    with io.open(packed_filename, mode='w', encoding='utf-8') as writer:
        pack(dataset, writer)
    logging.info("(2/3) - Tokenizing packed file to `%s`.", unpacked_filename)
    classpath = os.pathsep.join(requirements)

    cmd = 'javac -classpath %s %s' % (classpath, os.path.join(dir_name, 'TokenizerSplitter.java'))
    logging.info("Running `%s`", cmd)
    exit_status = os.system(cmd)
    if exit_status:
        sys.exit(exit_status)

    cmd = 'java -classpath %s TokenizerSplitter %s > %s' % (
        classpath, packed_filename, unpacked_filename)
    logging.info("Running `%s`\nMaluuba: The warnings below are normal.", cmd)
    exit_status = os.system(cmd)
    if exit_status:
        sys.exit(exit_status)

    os.remove(packed_filename)

    logging.info("(3/3) - Unpacking tokenized file to `%s`", output_path)
    with io.open(unpacked_filename, mode='r', encoding='utf-8') as packed:
        unpack(dataset, packed, output_path)

    os.remove(unpacked_filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser("NewsQA dataset parser")
    parser.add_argument("--cnn_stories", default='cnn_stories.tgz')
    parser.add_argument("--csv_dataset", default='newsqa-data-v1.csv')
    parser.add_argument("--combined_dataset", default='combined-newsqa-data-v1.csv')
    parser.add_argument("--output", default='newsqa-data-tokenized-v1.csv')
    args = parser.parse_args()

    tokenize(args.cnn_stories, args.csv_dataset, args.combined_dataset, args.output)
