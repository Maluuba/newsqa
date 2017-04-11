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
answer_char_ranges | Character based indices to answers in story_text. E.g. `196:228|196:202,217:228|None`. Answers from different crowdsourcers are separated by `|`, within those, multiple selections from the same crowdsourcer are separated by `,`.  `None` means the crowdsourcer thought there was no answer to the question in the story. The start is inclusive and the end is exclusive. The end may point to whitespace after a token.
is_answer_absent | Proportion of crowdsourcers that thought there was no answer to the question in the story.
is_question_bad | Proportion of crowdsourcers that thought the question does not make sense.
validated_answers | After crowdsourcing, we validated some answers when consensus was required. This shows how crowdsourcers voted during validation. E.g. `{"none": 1, "294:297": 2}` means that 1 crowdsourcer thought that none of the answers were good and 2 crowdsourcers thought that `294:297` was the best answer.

## PEP8
The code in this repository complies with PEP8 standards with a maximum line length of 99 characters.

## Requirements

* Download the CNN stories from [here][cnn_stories] to the maluuba/newsqa folder (for legal reasons, we can't automatically download these for you) 
* Download the questions and answers from [here][maluuba_newsqa_dl] to the maluuba/newsqa folder
* Extract the dowloaded tar.gz contents into the maluuba/newsqa folder (`tar -xzvf newsqa-data-v1.tar.gz`) (we'll automate this step in the future)
* Use Python 2
* (Optional) To tokenize the data and offset the character indices, you must get some JAR files. You can get the JAR files from [here][stanford_tagger]. You just need the [English option of version 3.6.0][stanford_zip_3.6.0]. Extract stanford-postagger-2015-12-09/stanford-postagger.jar and stanford-postagger-2015-12-09/lib/slf4j-api.jar to maluuba/newsqa
* Run `pip install --requirement requirements.txt`
* Run `python maluuba/newsqa/example.py --help` to see instructions
* (Testing) Make sure you do all above steps and run `python maluuba/newsqa/example.py && mv combined-newsqa-data-v1.csv maluuba/newsqa` before running tests 

## Package the Dataset
Run
```sh
python maluuba/newsqa/example.py
```

The file to check will be printed.

## Split the Dataset
To split the dataset into train, dev, and test, to match the paper run
```sh
python maluuba/newsqa/split_dataset.py
```

The file to check will be printed.

[cnn_stories]: http://cs.nyu.edu/~kcho/DMQA/
[maluuba_newsqa]: https://datasets.maluuba.com/NewsQA
[maluuba_newsqa_dl]: https://datasets.maluuba.com/NewsQA/dl
[stanford_tagger]: http://nlp.stanford.edu/software/tagger.html
[stanford_zip_3.6.0]: https://nlp.stanford.edu/software/stanford-postagger-2015-12-09.zip

## Tokenizing

To tokenize and split the dataset into train, dev, and test, to match the paper run 
```sh
python maluuba/newsqa/preprocess.py
```

## Legal

Notice:  CNN articles are used here by permission from The Cable News Network (CNN).  CNN does not waive any rights of ownership in its articles and materials.  CNN is not a partner of, nor does it endorse, Maluuba or its activities.

Terms: See `LICENSE.txt`.
