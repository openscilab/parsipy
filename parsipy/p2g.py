# -*- coding: utf-8 -*-
import pandas as pd
import word_stemmer

prefixes_dic = {
    'a': '',
    'an': '',
    'abē': 'ʾp̄y',
    'duš': 'dwš',
    'ham': 'hm',
    'hām': 'hʾm',
    'hu': 'hw',
    'jud': "ywdt'"
}
postfixes_dict = {
    'am': 'm',
    'ēd': "yt'",
    'ēnd': 'd',
    'ag': "k'",
    'āg': "'k",
    'agān': "kʾn",
    'an': 'n',
    'ān': "ʾn",
    'ānag': "ʾnk",
    'ār': "ʾl",
    'āwand': "ʾwnd",
    'bān': "bʾn",
    'bad': 'pt',
    'bed': 'pt',
    'dān': "dʾn",
    'ēn': "yn",
    'endag': "yndk",
    'estān': "stʾn",
    'gāh': "gʾs",
    'gānag': "kʾnk",
    'gar': "kl",
    'gār': "kʾl",
    'gēn': "kyn",
    'īg': "yk",
    'īgān': "ykʾn",
    'īh': "yh",
    'īha': "yhʾ",
    'īhā': "yh'",
    'išn': "šn'",
    'išt': "ʾšt'",
    'īzag': "yck",
    'om': "wm",
    'mand': "mnd",
    'ōmand': "'wmnd",
    'rōn': "lwn",
    'tar': "tl",
    'dar': "tl",
    'dār': "tʾl",
    'tom': "twm",
    'dom': "twm",
    'war': "wl",
    'wār': "wʾl",
    'zār': "cʾl",
    'īd': "yt",
}
transliteration_to_transcription_rules = {
    "t": ("t"),
    "š": ("š"),
    "č": ("c"),
    "p": ("p"),
    "f": ("f"),
    "s": ("s"),
    "h": ("s"),
    "l": ("l"),
    "k": ("k"),
    "z": ("z"),
    "r": ("r"),
    "n": ("n"),
    "w": ("w", "wb"),
    "m": ("m", "nb"),
    "y": ("y"),
    "g": ("g", "k"),
    "d": ("d"),
    "b": ("b", "p"),
    "a": ("h"),
    "ā": ("h"),
    "x": ("h"),
    "j": ("y"),
    "ē": ("i"),
    "e": ("i"),
    "i": ("i"),
    "c": ("c"),
    "u": ('w'),
    "ī": ('y'),
    "ō": ('w'),
    "ū": ('w'),
    "ǰ": ('c')
}

stems = pd.read_csv('stems_new.csv')


def oov_translit(word):
    first_letter = True
    new_word = ''
    for char in word:
        if first_letter:
            new_word += transliteration_to_transcription_rules[char][0]
        else:
            new_word += transliteration_to_transcription_rules[char][-1]
    return new_word


def get_transliteration(stem, post_fixes_list: list, pre_fixes_list: list):
    if post_fixes_list:
        post_fixes = ''.join(postfixes_dict[post_fix] for post_fix in reversed(post_fixes_list))
    else:
        post_fixes = ''

    if pre_fixes_list:
        pre_fixes = ''.join(prefixes_dic[pre_fix] for pre_fix in reversed(pre_fixes_list))
    else:
        pre_fixes = ''

    for stem_, transit_ in zip(stems['stem'], stems['stem_translit']):
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
