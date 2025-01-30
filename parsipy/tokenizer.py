# -*- coding: utf-8 -*-
def cleaning(text):
    return text.replace('*', '').replace('(w)', '').replace('[', '').replace(']', '').replace('<', '').replace('>',
                                                                                                               '')

def preprocess(sentence):
    sentence = sentence.replace('-', ' ')
    return cleaning(sentence)
