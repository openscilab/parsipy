# -*- coding: utf-8 -*-
from .data import EMISSION, TRANSITION
from .params import POSTaggerMethod, POS_MAPPING


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

                if word in self.emission_df.index:
                    emission_p = self.emission_df.loc[word, tag]
                elif smoothing:
                    emission_p = self.smoothing_const
                else:
                    emission_p = 0
                prob = emission_p * transition_p
                if prob >= max_prob:
                    max_prob = prob
                    best_tag = tag
            viterbi_list.append(best_tag)
        word_tag = list(zip(words, viterbi_list))
        return word_tag


class POSTagger:
    def __init__(self, model=POSTaggerMethod.RULE_BASED):
        self.model = model
        if model == POSTaggerMethod.RULE_BASED:
            self.tagger = POSTaggerRuleBased()
        else:
            pass

    def tag(self, sentence):
        if self.model == POSTaggerMethod.RULE_BASED:
            result = []
            for word, tag in self.tagger.viterbi(("<S> " + sentence).split(), smoothing=True):
                result.append({'text': word, 'POS': tag})
            return result[1:]
        else:
            pass


def run(sentence):
    tagger = POSTagger()
    return tagger.tag(sentence=sentence)