# -*- coding: utf-8 -*-
"""word stemmer."""
import copy
from .data import PREFIXES, POSTFIXES
from .data import ROOTS


def find_root(word):
    """
    Find the root of a word.

    :param word: Input word
    :type word: str
    :return: Dictionary containing the root of the word and the list of prefixes and postfixes
    """
    candidates = [{"stem": word, "post_fixes_list": [], "pre_fixes_list": []}]

    visited = list()
    while candidates:
        candidate = candidates.pop(0)
        if candidate['stem'] in ROOTS:
            return candidate

        if candidate not in visited:
            visited.append(candidate)

        for postfix in POSTFIXES.keys():
            if candidate['stem'].endswith(postfix):
                postfixes_list = candidate['post_fixes_list'].copy()
                prefix_list = candidate['pre_fixes_list'].copy()
                postfixes_list.append(postfix)
                stem = copy.deepcopy(candidate['stem'])[:-len(postfix)]
                new_candidate = {'stem': stem,
                                 'post_fixes_list': postfixes_list, 'pre_fixes_list': prefix_list}
                if new_candidate not in visited:
                    candidates.append(new_candidate)

        for prefix in PREFIXES.keys():
            if candidate['stem'].startswith(prefix):
                postfixes_list = candidate['post_fixes_list'].copy()
                prefix_list = candidate['pre_fixes_list'].copy()
                prefix_list.append(prefix)
                stem = copy.deepcopy(candidate['stem'])[len(prefix):]
                new_candidate = {'stem': stem,
                                 'post_fixes_list': postfixes_list, 'pre_fixes_list': prefix_list}
                if new_candidate not in visited:
                    candidates.append(new_candidate)
    return {'stem': word, 'post_fixes_list': [], 'pre_fixes_list': []}


def run(sentence):
    """
    Run the word stemmer on a sentence.

    :param sentence: Input sentence
    :type sentence: str
    :return: List of dictionaries containing the root of each word in the sentence
    """
    result = []
    for word in sentence.split():
        roots = find_root(word)
        data = {'text': word,
                'stem': roots['stem']}
        result.append(data)
    return result
