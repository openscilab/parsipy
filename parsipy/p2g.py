# -*- coding: utf-8 -*-
import pandas as pd
import word_stemmer
from .data import PREFIXES, POSTFIXES
from .data import TRANSLITERATION_TO_TRANSCRIPTION_RULES
from .data import STEMS




def oov_translit(word):
    first_letter = True
    new_word = ''
    for char in word:
        if first_letter:
            new_word += TRANSLITERATION_TO_TRANSCRIPTION_RULES[char][0]
        else:
            new_word += TRANSLITERATION_TO_TRANSCRIPTION_RULES[char][-1]
    return new_word


def get_transliteration(stem, post_fixes_list: list, pre_fixes_list: list):
    if post_fixes_list:
        post_fixes = ''.join(POSTFIXES[post_fix] for post_fix in reversed(post_fixes_list))
    else:
        post_fixes = ''

    if pre_fixes_list:
        pre_fixes = ''.join(PREFIXES[pre_fix] for pre_fix in reversed(pre_fixes_list))
    else:
        pre_fixes = ''

    for stem_, transit_ in zip(STEMS['stem'], STEMS['stem_translit']):
        if stem_ == stem:
            return str(pre_fixes) + str(transit_) + str(post_fixes)
    # Return None if no match is found to ensure consistent function behavior
    return oov_translit(stem)


def run(sentence: str):
    my_list = []
    for word in sentence.split():
        roots = word_stemmer.find_root(word)
        data = {'text': word, 'transliteration': get_transliteration(**roots)}
        my_list.append(data)
    return my_list


if __name__ == '__main__':
    print(run('ruwān ī xwēš andar ayād dār nām ī xwēš rāy xwēš-kārīh '))
