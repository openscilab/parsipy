<div align="center">
    <h1>ParsiPy: NLP Toolkit for Historical Persian Texts in Python</h1>
    <br/>
    <a href="https://badge.fury.io/py/parsipy"><img src="https://badge.fury.io/py/parsipy.svg" alt="PyPI version"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/built%20with-Python3-green.svg" alt="built with Python3"></a>
    <a href="https://github.com/openscilab/parsipy"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/openscilab/parsipy"></a>
</div>

----------


## Overview
<p align="justify">
ParsiPy is an NLP toolkit designed for analyzing historical Persian texts, including languages like Parsig (Pahlavi). It provides essential modules such as lemmatization, POS tagging, tokenization, and phoneme-to-grapheme conversion, making it a valuable resource for researchers working with low-resource languages. Beyond its practical applications, ParsiPy serves as a model for developing NLP tools tailored to linguistically rich yet underrepresented languages.
</p>

<table>
    <tr>
        <td align="center">PyPI Counter</td>
        <td align="center">
            <a href="https://pepy.tech/projects/parsipy">
                <img src="https://static.pepy.tech/badge/parsipy">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">Github Stars</td>
        <td align="center">
            <a href="https://github.com/openscilab/parsipy">
                <img src="https://img.shields.io/github/stars/openscilab/parsipy.svg?style=social&label=Stars">
            </a>
        </td>
    </tr>
</table>
<table>
    <tr> 
        <td align="center">Branch</td>
        <td align="center">main</td>
        <td align="center">dev</td>
    </tr>
    <tr>
        <td align="center">CI</td>
        <td align="center">
            <img src="https://github.com/openscilab/parsipy/actions/workflows/test.yml/badge.svg?branch=main">
        </td>
        <td align="center">
            <img src="https://github.com/openscilab/parsipy/actions/workflows/test.yml/badge.svg?branch=dev">
            </td>
    </tr>
</table>


## Installation

### PyPI
- Check [Python Packaging User Guide](https://packaging.python.org/installing/)
- Run `pip install parsipy==0.1`
### Source code
- Download [Version 0.1](https://github.com/openscilab/parsipy/archive/v0.1.zip) or [Latest Source](https://github.com/openscilab/parsipy/archive/dev.zip)
- Run `pip install .`

## Usage
To use ParsiPy's modules for analyzing texts in the Pahlavi language, you need to input your text in phonetic form.  
To simplify the process, we have developed a pipeline module that works as follows.

### Pipeline
In the following example, we use a passage from an ancient Parsig text containing advice for people at that time.
Its rough English translation is: *"Forget what is gone and do not worry about what has not yet come."* [1]  

You can easily apply tokenization, lemmatization, POS tagging, and phoneme-to-grapheme conversion to this text using the following code:

```pycon
>>> from parsipy import pipeline, Task
>>> result = pipeline(sentence='ān uzīd frāmōš kun ud ān nē mad ēstēd rāy tēmār bēš ma bar',
                      tasks=[Task.TOKENIZER, Task.LEMMA, Task.POS, Task.P2T])
```

The result is a dictionary containing the outputs of all requested tasks:

```json
{
    "tokenizer": [
        {"id": 0, "text": "ān"},
        {"id": 1, "text": "uzīd"},
        {"id": 2, "text": "frāmōš"},
        {"id": 3, "text": "kun"},
        {"id": 4, "text": "ud"},
        {"id": 5, "text": "ān"},
        {"id": 6, "text": "nē"},
        {"id": 7, "text": "mad"},
        {"id": 8, "text": "ēstēd"},
        {"id": 9, "text": "rāy"},
        {"id": 10, "text": "tēmār"},
        {"id": 11, "text": "bēš"},
        {"id": 12, "text": "ma"},
        {"id": 13, "text": "bar"}
    ],
    "lemma": [
        {"stem": "ān", "text": "ān"},
        {"stem": "uzīd", "text": "uzīd"},
        {"stem": "frāmōš", "text": "frāmōš"},
        {"stem": "kun", "text": "kun"},
        {"stem": "ud", "text": "ud"},
        {"stem": "ān", "text": "ān"},
        {"stem": "nē", "text": "nē"},
        {"stem": "mad", "text": "mad"},
        {"stem": "ēst", "text": "ēstēd"},
        {"stem": "rāy", "text": "rāy"},
        {"stem": "tēmār", "text": "tēmār"},
        {"stem": "bēš", "text": "bēš"},
        {"stem": "ma", "text": "ma"},
        {"stem": "bar", "text": "bar"}
    ],

    "POS": [
        {"POS": "DET", "text": "ān"},
        {"POS": "N", "text": "uzīd"},
        {"POS": "N", "text": "frāmōš"},
        {"POS": "V", "text": "kun"},
        {"POS": "CONJ", "text": "ud"},
        {"POS": "DET", "text": "ān"},
        {"POS": "ADV", "text": "nē"},
        {"POS": "V", "text": "mad"},
        {"POS": "V", "text": "ēstēd"},
        {"POS": "POST", "text": "rāy"},
        {"POS": "N", "text": "tēmār"},
        {"POS": "N", "text": "bēš"},
        {"POS": "ADV", "text": "ma"},
        {"POS": "N", "text": "bar"}
    ],
    "P2T": [
        {"text": "ān", "transliteration": "ZK"},
        {"text": "uzīd", "transliteration": "ʾwcyt"},
        {"text": "frāmōš", "transliteration": "plʾmwš"},
        {"text": "kun", "transliteration": "OḆYDWNt͟y"},
        {"text": "ud", "transliteration": "W"},
        {"text": "ān", "transliteration": "ZK"},
        {"text": "nē", "transliteration": "LA"},
        {"text": "mad", "transliteration": "mt"},
        {"text": "ēstēd", "transliteration": "YKOYMWyt'"},
        {"text": "rāy", "transliteration": "lʾd"},
        {"text": "tēmār", "transliteration": "tymʾl"},
        {"text": "bēš", "transliteration": "byš"},
        {"text": "ma", "transliteration": "AL"},
        {"text": "bar", "transliteration": "YḆLWN"}
    ]
}
```

Below is a brief explanation of each task:

#### Tokenization
This module splits a sentence into individual tokens, making it easier to process each word separately. Tokenization is a crucial first step for many NLP tasks.

#### Lemmatization
Lemmatization reduces words to their base or root forms, removing prefixes and suffixes. This is useful for standardizing different word variations.

#### POS
This module assigns a part-of-speech (POS) tag to each word in a sentence based on its grammatical role. The output provides essential grammatical information for further text analysis.

#### P2T
Since there is no widely accepted Unicode representation for the original Pahlavi script, digital texts are often written in a phonetic form.
This module maps phonetic representations to their transliteration which is a middle-form between phonetic and their original characters.
We also present a tool for converting the transliteration into the original text format.

For converting transliteration to Parsig font, you can use [this exe file and font](https://drive.google.com/drive/folders/1hja6fXt9zZHBs1Hf8YFZl8utygVqbsyd?usp=sharing) in Windows.

## Issues & bug reports

Just fill an issue and describe it. We'll check it ASAP! or send an email to [parsipy@openscilab.com](mailto:parsipy@openscilab.com "parsipy@openscilab.com"). 

- Please complete the issue template


## References
1- گشتاسب, فرزانه, and حاجی پور. "توصیف و تبیین ماهیت عدالت خسرو انوشیروان در متون فارسی و جستجوی پیشینه آن در متون فارسی میانه." (فصلنامه مطالعات تاریخ فرهنگی) پژوهشنامه انجمن ایرانی تاریخ 14.53 (2022): 101-125.

## Show your support


### Star this repo

Give a ⭐️ if this project helped you!

### Donate to our project
If you do like our project and we hope that you do, can you please support us? Our project is not and is never going to be working for profit. We need the money just so we can continue doing what we do ;-) .			

<a href="https://openscilab.com/#donation" target="_blank"><img src="https://github.com/openscilab/parsipy/raw/main/otherfiles/donation.png" height="90px" width="270px" alt="ParsiPy Donation"></a>

