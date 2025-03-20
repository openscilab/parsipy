# -*- coding: utf-8 -*-
"""transliteration."""
from .word_stemmer import find_root
from .data import PREFIXES, POSTFIXES
from .data import TRANSLITERATION_TO_TRANSCRIPTION_RULES
from .data import STEMS


def oov_transliterate(word):
    """
    Transliterates a word that is not in the STEMS dictionary.

    :param word: word to be transliterated
    :type word: str
    :return: str
    """
    first_letter = True
    new_word = ''
    for char in word:
        if first_letter:
            new_word += TRANSLITERATION_TO_TRANSCRIPTION_RULES[char][0]
        else:
            new_word += TRANSLITERATION_TO_TRANSCRIPTION_RULES[char][-1]
    return new_word


def get_transliteration(stem, post_fixes_list, pre_fixes_list):
    """
    Transliteration of a word given its stem and affixes.

    :param stem: word stem
    :type stem: str
    :param post_fixes_list: list of post-fixes
    :type post_fixes_list: list
    :param pre_fixes_list: list of pre-fixes
    :type pre_fixes_list: list
    :return: str
    """
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
    return oov_transliterate(stem)


def run(sentence):
    """
    Run the transliteration on a sentence.

    :param sentence: sentence to be processed
    :type sentence: str
    :return: list
    """
    result = []
    for word in sentence.split():
        roots = find_root(word)
        data = {'text': word,
                'transliteration': get_transliteration(**roots)}
        result.append(data)
    return result
