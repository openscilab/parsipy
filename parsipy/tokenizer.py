# -*- coding: utf-8 -*-
def cleaning(text):
    return text.replace('*', '').replace('(w)', '').replace('[', '').replace(']', '').replace('<', '').replace('>',
                                                                                                               '')

def preprocess(sentence):
    sentence = sentence.replace('-', ' ')
    return cleaning(sentence)


def run(sentence):
    sentence = preprocess(sentence)
    result = []
    for index, word in enumerate(sentence.split()):
        data = {'id': index, 'text': word}
        result.append(data)
    return result
