# -*- coding: utf-8 -*-
import pandas as pd
from .params import EMISSION, TRANSITION, POS_MAPPING


class POSTaggerRuleBased:
    def __init__(self, smoothing_const=0.0001):
        self.smoothing_const = smoothing_const
        self.trans_df = TRANSITION
        self.emission_df = EMISSION
        self.tagset = sorted(set(POS_MAPPING.values()))

    def viterbi(self, words, smoothing=True):
        viterbi_list = []
        for key, word in enumerate(words):
            if word.endswith('īhā'):
                viterbi_list.append('ADV')
                continue
            max_prob = -1
            best_tag = None
            for tag in self.tagset:
                if key == 0:
                    transition_p = self.trans_df.loc['<S>', tag]
                else:
                    last_state = viterbi_list[-1]
                    transition_p = self.trans_df.loc[last_state, tag]

                if word in self.vocab:
                    emission_p = self.emission_df.loc[word, tag]
                elif smoothing:
                    emission_p = self.smoothing_const
                else:
                    emission_p = 0
                prob = emission_p * transition_p
                if prob >= max_prob:
                    max_prob = prob
                    best_tag = tag
            #     print(f"word: {word}, tag: {tag}, emission_p: {emission_p}, transition_p: {transition_p}, prob: {prob}, best_tag: {best_tag}")
            viterbi_list.append(best_tag)
        word_tag = list(zip(words, viterbi_list))
        return word_tag
