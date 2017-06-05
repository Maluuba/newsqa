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
answer_token_ranges | word based indices to answers in story_text. E.g. `196:202,217:228`. Multiple selections from the same answer are separated by `,`. The start is inclusive and the end is exclusive. The end may point to whitespace after a token.

## PEP8
The code in this repository complies with PEP8 standards with a maximum line length of 99 characters.

## Requirements

* Download the CNN stories from [here][cnn_stories] to the maluuba/newsqa folder (for legal reasons, we can't automatically download these for you) 
* Download the questions and answers from [here][maluuba_newsqa_dl] to the maluuba/newsqa folder
* Extract the dowloaded tar.gz contents into the maluuba/newsqa folder (`tar -xzvf newsqa-data-v1.tar.gz`) (we'll automate this step in the future)
* Use Python 2 (Python 2 code was originally used to handle the stories and they got encoded strangely)
* (Optional - Tokenization) To tokenize the data, you must install a JDK (Java Development Kit) so that you can compile and run Java code.
* (Optional - Tokenization) To tokenize the data, you must get some JAR files. You can get the JAR files from [here][stanford_tagger]. You just need the [English option of version 3.6.0][stanford_zip_3.6.0]. Extract stanford-postagger-2015-12-09/stanford-postagger.jar and stanford-postagger-2015-12-09/lib/slf4j-api.jar to maluuba/newsqa
* Run `pip install --requirement requirements.txt`

[cnn_stories]: http://cs.nyu.edu/~kcho/DMQA/
[maluuba_newsqa]: https://datasets.maluuba.com/NewsQA
[maluuba_newsqa_dl]: https://datasets.maluuba.com/NewsQA/dl
[stanford_tagger]: http://nlp.stanford.edu/software/tagger.html
[stanford_zip_3.6.0]: https://nlp.stanford.edu/software/stanford-postagger-2015-12-09.zip

## Package the Dataset

## Tokenize and Split
To tokenize and split the dataset into train, dev, and test, to match the paper run 
```sh
python maluuba/newsqa/data_generator.py
```

The warnings from the tokenizer are normal.

## Legal

Notice:  CNN articles are used here by permission from The Cable News Network (CNN).  CNN does not waive any rights of ownership in its articles and materials.  CNN is not a partner of, nor does it endorse, Maluuba or its activities.

Terms: See `LICENSE.txt`.
