# -*- coding: utf-8 -*-
import copy
from .data import PREFIXES, POSTFIXES
from .data import ROOTS

def find_root(word):
    def is_goal(stem):
        if stem in ROOTS:
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
