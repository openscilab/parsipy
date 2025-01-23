# -*- coding: utf-8 -*-
import pandas as pd
import copy

df_ = list(pd.read_json('roots.json')['data'])

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


def find_root(word: str):
    def is_goal(stem):
        if stem in df_:
            return True
        return False

    candidates = [{"stem": word, "post_fixes_list": [], "pre_fixes_list": []}]

    visited = list()
    while candidates:
        candidate = candidates.pop(0)
        if is_goal(candidate['stem']):
            return candidate

        if candidate not in visited:
            visited.append(candidate)

        for postfix in postfixes_dict.keys():
            if candidate['stem'].endswith(postfix):
                postfixes_list = candidate['post_fixes_list'].copy()
                prefix_list = candidate['pre_fixes_list'].copy()
                postfixes_list.append(postfix)
                stem = copy.deepcopy(candidate['stem'])[:-len(postfix)]
                new_candidate = {'stem': stem,
                                 'post_fixes_list': postfixes_list, 'pre_fixes_list': prefix_list}
                if new_candidate not in visited:
                    candidates.append(new_candidate)

        for prefix in prefixes_dic.keys():
            if candidate['stem'].startswith(prefix):
                postfixes_list = candidate['post_fixes_list'].copy()
                prefix_list = candidate['pre_fixes_list'].copy()
                prefix_list.append(prefix)
                stem = copy.deepcopy(candidate['stem'])[len(prefix):]
                new_candidate = {'stem': stem,
                                 'post_fixes_list': postfixes_list, 'pre_fixes_list': prefix_list}
                if new_candidate not in visited:
                    candidates.append(new_candidate)
    ## none
    return {'stem': word, 'post_fixes_list': [], 'pre_fixes_list': []}


def run(sentence: str):
    my_list = []
    for word in sentence.split():
        roots = find_root(word)
        data = {'text': word, 'stem': roots['stem']}
        my_list.append(data)
    return my_list


if __name__ == '__main__':
    print(run('ruwān ī xwēš andar ayād dār nām ī xwēš rāy xwēš-kārīh '))
