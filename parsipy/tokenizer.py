# -*- coding: utf-8 -*-
"""tokenizer."""


def cleaning(text):
    """
    Clean the text from special characters.

    :param text: input text
    :type text: str
    :return: cleaned text
    """
    return text.replace('*', '')\
        .replace('(w)', '')\
        .replace('[', '')\
        .replace(']', '')\
        .replace('<', '')\
        .replace('>', '')


def preprocess(sentence):
    """
    Preprocess the sentence.

    :param sentence: input sentence
    :type sentence: str
    :return: preprocessed sentence
    """
    sentence = sentence.replace('-', ' ')
    return cleaning(sentence)


def run(sentence):
    """
    Tokenize the sentence.

    :param sentence: input sentence
    :type sentence: str
    :return: list of tokenized words
    """
    sentence = preprocess(sentence)
    result = []
    for index, word in enumerate(sentence.split()):
        data = {'id': index,
                'text': word}
        result.append(data)
    return result
