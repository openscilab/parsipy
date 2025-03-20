# -*- coding: utf-8 -*-
"""data."""
import os
import pandas as pd


def get_data_path(data_name):
    """
    Return data file path.

    :param data_name: data file name
    :type data_name: str
    :return: direct path to data file
    """
    cd, _ = os.path.split(__file__)
    return os.path.join(cd, "files", data_name)


ROOTS = list(pd.read_json(get_data_path('roots.json'))['data'])

STEMS = pd.read_csv(get_data_path('stems.csv'))

EMISSION = pd.read_csv(get_data_path("emission.csv"), index_col=0)

TRANSITION = pd.read_csv(get_data_path("transition.csv"), index_col=0)


PREFIXES = {
    'a': '',
    'an': '',
    'abē': 'ʾp̄y',
    'duš': 'dwš',
    'ham': 'hm',
    'hām': 'hʾm',
    'hu': 'hw',
    'jud': "ywdt'",
}

POSTFIXES = {
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


TRANSLITERATION_TO_TRANSCRIPTION_RULES = {
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
    "ǰ": ('c'),
}
