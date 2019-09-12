# Maluuba NewsQA
Tools for using Maluuba's news questions and answer data. The code in the repo is used to compile the dataset since it cannot be made directly available due to legal reasons.

You can find more information with stats on the dataset [here][maluuba_newsqa].

## Data Description
We originally only compiled to CSV but now we also build a JSON file.

### JSON
Here is an example of how the data in `combined-newsqa-data-v1.json` will look:
```json
{
    "data": [
        {
            "storyId": "./contoso/stories/2e1d4",
            "text": "Hyrule (Contoso) -- Tingle, Tingle! Kooloo-Limpah! ...These are the magic words that Tingle created himself. Don't steal them!",
            "type": "train",
            "questions": [
                {
                    "q": "What should you not do with Tingle's magic words?",
                    "consensus": {
                        "s": 115,
                        "e": 125
                    },
                    "isAnswerAbsent": 0.25,
                    "isQuestionBad": 0.25,
                    "answers": [
                        {
                            "sourcerAnswers": [
                                {
                                    "s": 115,
                                    "e": 125
                                }
                            ]
                        },
                        {
                            "sourcerAnswers": [
                                {
                                    "noAnswer": true
                                }
                            ]
                        }
                    ],
                    "validatedAnswers": [
                        {
                            "s": 115,
                            "e": 125,
                            "count": 2
                        },
                        {
                            "noAnswer": true,
                            "count": 1
                        },
                        {
                            "badQuestion": true,
                            "count": 1
                        }
                    ]
                }
            ]
        }
    ],
    "version": "1"
}
```
Explanation:

Field | Description
--- | ---
data | A list of the data for the dataset.
storyId | The identifier for the story. Comes from the member name in the CNN stories package.   
text | The text for the story.
type | The type of data this should be used for. Will be "train", "dev", or "test".  
questions | The questions about the story.
q | A question about the story.
consensus | The consensus answer. Use this field to pick the best continuous answer span from the text. If you want to know about a question having multiple answers in the text then you can use the more detailed "answers" and "validatedAnswers". The object can have start and end positions like in the example above or can be `{"badQuestion": true}` or `{"noAnswer": true}`. Note that there is only one consensus answer since it's based on the majority agreement of the crowdsourcers.
isAnswerAbsent | Proportion of crowdsourcers that said there was no answer to the question in the story.
isQuestionBad | Proportion of crowdsourcers that said the question does not make sense.
version | The version string for the dataset.

Explanation of the answer fields:

Field | Description
--- | ---
answers | The answers from various crowdsourcers.
sourcerAnswers| The answer provided from one crowdsourcer.
validatedAnswers | The answers from the validators.
s | The first character of the answer in "text" (inclusive).
e | The last character of the answer in "text" (exclusive).
noAnswer | The crowdsourcer said that there was no answer to the question in the text.
badQuestion | The validator said that the question did not make sense.
count | The number of validators that agreed with this answer.

### CSV
This section describes the CSV formatted files for the dataset. We originally only compiled to CSV.

The combined dataset in the .csv file is made of several columns to show the story text and the derived answers from several crowdsourcers.

Column Name | Description
--- | ---
story_id | The identifier for the story. Comes from the member name in the CNN stories package.
story_text | The text for the story.
question | A question about the story.
answer_char_ranges | (in combined-newsqa-data-*.csv) The raw data collected for character based indices to answers in story_text. E.g. `196:228\|196:202,217:228\|None`. Answers from different crowdsourcers are separated by `\|`, within those, multiple selections from the same crowdsourcer are separated by `,`.  `None` means the crowdsourcer thought there was no answer to the question in the story. The start is inclusive and the end is exclusive. The end may point to whitespace after a token. | Note that the `\` isn't actually in the data, it's just in this README so that it displays nicely on GitHub.
answer_token_ranges | (in newsqa-data-tokenized-*.csv) Word based indices to answers in story_text. E.g. `196:202,217:228`. Multiple selections from the same answer are separated by `,`. The start is inclusive and the end is exclusive. The end may point to whitespace after a token.

There are some other fields in combined-newsqa-data-*.csv for raw data collected when crowdsourcing such as the validation of collected data.

## Requirements

Run either the Docker steps or do the manual set up.

The dataset for character based indices will be the `combined-newsqa-data-*.csv` file in the root.

The dataset for token based indices will be `maluuba/newsqa/newsqa-data-tokenized-*.csv`.

### (Recommended) Docker Set Up
These steps handle packaging the dataset and running the tests.

* Clone this repo.
* Download the tar.gz file for the questions and answers from [here][maluuba_newsqa_dl] to the maluuba/newsqa folder. No need to extract anything.
* Download the CNN stories from [here][cnn_stories] to the maluuba/newsqa folder (for legal and technical reasons, we can't distribute this to you).
* In the root of this repo, run:
```bash
docker build -t maluuba/newsqa .
docker run --rm -it -v ${PWD}:/usr/src/newsqa --name newsqa maluuba/newsqa
```

You now have the datasets.  See `combined-newsqa-data-*.json`, `combined-newsqa-data-*.csv`, or `maluuba/newsqa/newsqa-data-tokenized-*.csv`.

#### Tokenize and Split
If you want to tokenize and split the data into train, dev, and test, to match the paper run, then you must get "into" the container and run the packaging command:
```bash
docker run --rm -it -v ${PWD}:/usr/src/newsqa --name newsqa maluuba/newsqa /bin/bash --login -c 'python maluuba/newsqa/data_generator.py'
```
The warnings from the tokenizer are normal.

#### Troubleshooting Docker Set Up
If you run into issues such as the tokenization not unpacking, then you may need to give Docker at least 4GB of memory.

### Manual Set Up
* Clone this repo.
* Download the tar.gz file for the questions and answers from [here][maluuba_newsqa_dl] to the maluuba/newsqa folder. No need to extract anything.
* Download the CNN stories from [here][cnn_stories] to the maluuba/newsqa folder (for legal and technical reasons, we can't distribute this to you).
* Use Python 2.7 to package the dataset (Python 2.7 was originally used to handle the stories and they got encoded strangely - once the dataset is packaged by these scripts, you should be able to load the files with whatever tools you'd like). You can create a [Conda][conda] environment like so:
```bash
conda create --name newsqa python=2.7 "pandas>=0.19.2"
```
* Install the requirements in your environment:
```bash
conda activate newsqa && pip install --requirement requirements.txt
```
* (Optional - Tokenization) To tokenize the data, you must install a JDK (Java Development Kit) so that you can compile and run Java code.
* (Optional - Tokenization) To tokenize the data, you must get some JAR files. We use some libraries from [Stanford][stanford_tagger]. You just need to put the [English option of version 3.6.0][stanford_zip_3.6.0] in the maluuba/newsqa folder.

#### Package the Dataset

##### Tokenize and Split
To tokenize and split the dataset into train, dev, and test, to match the paper run:
```sh
python maluuba/newsqa/data_generator.py
```

The warnings from the tokenizer are normal.

#### Testing
To make sure that everything is extracted right, run
```bash
python -m unittest discover .
```
All tests should pass.

[conda]: https://conda.io/miniconda.html
[cnn_stories]: http://cs.nyu.edu/~kcho/DMQA/
[maluuba_newsqa]: https://www.microsoft.com/en-us/research/project/newsqa-dataset
[maluuba_newsqa_dl]: https://msropendata.com/datasets/939b1042-6402-4697-9c15-7a28de7e1321
[stanford_tagger]: http://nlp.stanford.edu/software/tagger.html
[stanford_zip_3.6.0]: https://nlp.stanford.edu/software/stanford-postagger-2015-12-09.zip

## PEP8
The code in this repository complies with PEP8 standards with a maximum line length of 99 characters.

## Legal

Notice:  CNN articles are used here by permission from The Cable News Network (CNN).  CNN does not waive any rights of ownership in its articles and materials.  CNN is not a partner of, nor does it endorse, Maluuba or its activities.

Terms: See `LICENSE.txt`.
