# Maluuba NewsQA
Tools for using Maluuba's news questions and answer data.

You can find more information about the dataset [here][maluuba_newsqa].

## Data Description

The combined dataset is made of several columns to show the story text and the derived answers from several crowdsourcers.

Column Name | Description
--- | ---
story_id | The identifier for the story. Comes from the member name in the CNN stories package.
story_text | The text for the story.
question | A question about the story.
answer_char_ranges | (in combined-newsqa-data-*.csv) The raw data collected for character based indices to answers in story_text. E.g. `196:228|196:202,217:228|None`. Answers from different crowdsourcers are separated by `|`, within those, multiple selections from the same crowdsourcer are separated by `,`.  `None` means the crowdsourcer thought there was no answer to the question in the story. The start is inclusive and the end is exclusive. The end may point to whitespace after a token.
answer_token_ranges | (in newsqa-data-tokenized-*.csv) Word based indices to answers in story_text. E.g. `196:202,217:228`. Multiple selections from the same answer are separated by `,`. The start is inclusive and the end is exclusive. The end may point to whitespace after a token.

There are some other fields in combined-newsqa-data-*.csv for raw data collected when crowdsourcing such as the validation of collected data.

## Requirements

* Download the questions and answers from [here][maluuba_newsqa_dl] to the maluuba/newsqa folder. No need to extract anything.
* Run either the Docker steps which handle everything or do the manual set up.

The dataset for token based indices will be the `combined-newsqa-data-*.csv` file in the root.

The dataset for character based indices will be `maluuba/newsqa/newsqa-data-tokenized-*.csv`.

### Docker Set Up
These steps handle packaging the dataset and running the tests.

In the root of this repo, run:
```bash
docker build -t maluuba/newsqa .
docker run --rm -it -v ${PWD}:/usr/src/newsqa --name newsqa maluuba/newsqa
```

### Manual Set Up
* Download the CNN stories from [here][cnn_stories] to the maluuba/newsqa folder (for legal reasons, we can't automatically download these for you).
* Use Python 2.7 to package the dataset (Python 2.7 was originally used to handle the stories and they got encoded strangely - once the dataset is packaged by these scripts, you should be able to load the files with whatever tools you'd like) You can create a [Conda][conda] environment like so:
```bash
conda create --name newsqa python=2.7 "pandas>=0.19.2"
```
* Install the requirements in your environment:
```bash
pip install --requirement requirements.txt
```
* (Optional - Tokenization) To tokenize the data, you must install a JDK (Java Development Kit) so that you can compile and run Java code.
* (Optional - Tokenization) To tokenize the data, you must get some JAR files. We use some libraries from [Stanford][stanford_tagger]. You just need to put the [English option of version 3.6.0][stanford_zip_3.6.0] in the maluuba/newsqa folder.

[conda]: https://conda.io/miniconda.html
[cnn_stories]: http://cs.nyu.edu/~kcho/DMQA/
[maluuba_newsqa]: https://datasets.maluuba.com/NewsQA
[maluuba_newsqa_dl]: https://datasets.maluuba.com/NewsQA/dl
[stanford_tagger]: http://nlp.stanford.edu/software/tagger.html
[stanford_zip_3.6.0]: https://nlp.stanford.edu/software/stanford-postagger-2015-12-09.zip

## Package the Dataset

### Tokenize and Split
To tokenize and split the dataset into train, dev, and test, to match the paper run 
```sh
python maluuba/newsqa/data_generator.py
```

The warnings from the tokenizer are normal.

## Testing
To make sure that everything is extracted right, run
```bash
python -m unittest discover .
```
All tests should pass.

## PEP8
The code in this repository complies with PEP8 standards with a maximum line length of 99 characters.

## Legal

Notice:  CNN articles are used here by permission from The Cable News Network (CNN).  CNN does not waive any rights of ownership in its articles and materials.  CNN is not a partner of, nor does it endorse, Maluuba or its activities.

Terms: See `LICENSE.txt`.
