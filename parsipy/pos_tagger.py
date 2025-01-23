# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from collections import defaultdict

class ParsigPOSLoader:
    def __init__(self, dataset_file_name, test_portion=0.1):
        self.test_portion = test_portion
        self.MPTextWord = None
        self.MPTextWord2 = None
        self.train_set = None
        self.test_set = None
        self.train_words_tag = None
        self.test_words_tag = None
        self.parsig_sentences = None
        self.dataset_file_name = dataset_file_name
        self.read_data()
        self.pos_mapping()
        self.set_all_word_tag_pairs()
        if test_portion > 0:
            self.split_train_val_test_set()
        else:
            self.train_set = self.parsig_sentences

    def clean_text(self, df):
        chars_to_remove = ['*', '[', ']', '<', '>', '-']
        cleaned_df = df.copy()
        cleaned_df['Transcription'] = cleaned_df['Transcription'].apply(
            lambda x: re.sub(r'\([^()]*\)', '', str(x))
        )
        cleaned_df['Transcription'] = cleaned_df['Transcription'].apply(
            lambda x: re.sub(r'\.$', ' <S>', str(x))
        )
        cleaned_df['Transcription'] = cleaned_df['Transcription'].apply(
            lambda x: ''.join(char for char in str(x) if char not in chars_to_remove)
        )
        return cleaned_df

    def read_data(self):
        xl = pd.ExcelFile(self.dataset_file_name)
        self.MPTextWord = pd.read_excel(xl, sheet_name="MPTextWord")
        self.MPTextWord2 = pd.read_excel(xl, sheet_name="MPTextWord 2")
        self.MPTextWord = self.clean_text(self.MPTextWord)
        self.MPTextWord2 = self.clean_text(self.MPTextWord2)
        self.MPTextWord = self.MPTextWord.drop(['Pronunciation', 'EnTranslation', 'Transliteration', 'Translation', 'WordRoot'], axis=1)
        self.MPTextWord2 = self.MPTextWord2.drop(['Pronunciation', 'EnTranslation', 'Transliteration', 'Translation', 'WordRoot'], axis=1)

    def pos_mapping(self):
        pos_mapping = {
            10: 'N',     # Noun
            20: 'ADJ',   # ADJective
            21: 'DET',   # DETerminer
            30: 'V',     # Verb
            40: 'ADV',   # ADVerb
            50: 'PRON',  # PRONoun
            60: 'PREP',  # PREPosition
            61: 'POST',  # POSTposition
            63: 'CONJ',  # CONJunction
            64: 'EZ',    # EZafeh
            70: 'NUM',   # NUMber
            80: 'PART'   # PARTicle
        }

        self.MPTextWord['Pos'] = self.MPTextWord['WordCategoryCode'].map(pos_mapping)
        self.MPTextWord['Pos'] = self.MPTextWord['WordCategoryCode'].map(pos_mapping).fillna('Unknown')

        self.MPTextWord2['Pos'] = self.MPTextWord2['WordCategoryCode'].map(pos_mapping)
        self.MPTextWord2['Pos'] = self.MPTextWord2['WordCategoryCode'].map(pos_mapping).fillna('Unknown')

    def get_word_tag(self, df):
        """
        sentences will be in format:
        [
           [('word1', 'POS'), ('word2', 'POS'), ...],
            ...
        ]
        """
        sentences = []
        for ref, group in df.groupby('Reference2'):
            sentence_pairs = list(zip(group['Transcription'], group['Pos']))
            sentences.append(sentence_pairs)
        return sentences

    def set_all_word_tag_pairs(self):
        MPTextWord_POS = self.get_word_tag(self.MPTextWord)
        MPTextWord2_POS = self.get_word_tag(self.MPTextWord2)
        self.parsig_sentences = MPTextWord_POS + MPTextWord2_POS
        self.parsig_sentences = [[('<S>', '<S>')] + sent for sent in self.parsig_sentences]

    def split_train_val_test_set(self):
        self.train_set, self.test_set = train_test_split(self.parsig_sentences, test_size=self.test_portion, random_state=100)
        return self.train_set, self.test_set

    def extract_all_words_tag(self):
        self.train_words_tag = [word_tag for record in self.train_set for word_tag in record]
        if self.test_set is not None:
            self.test_words_tag = [word_tag for record in self.test_set for word_tag in record]
        return self.train_words_tag, self.test_words_tag

    def set_vocab_and_tagset(self):
        vocab = set([word_tag[0] for word_tag in self.train_words_tag])
        tagset = sorted(list(set([pair[1] for pair in self.train_words_tag])))
        return vocab, tagset


class POSTagger:
    def __init__(self, dataset_loader, smoothing_const=0.0001, Training=True):
        self.emission_file_name = 'emission_parsig.csv'
        self.transition_file_name = 'transition_parsig.csv'
        self.smoothing_const = smoothing_const
        if Training:
            self.train_set, self.test_set = dataset_loader.split_train_val_test_set()
            self.test_run_base = [tup for sent in self.test_set for tup in sent]
            self.test_tagged_words = [tup[0] for sent in self.test_set for tup in sent]
        self.train_words_tag, self.test_words_tag = dataset_loader.extract_all_words_tag()
        self.vocab, self.tagset = dataset_loader.set_vocab_and_tagset()
        self.t = len(self.tagset)
        self.v = len(self.vocab)
        self.parsig_sentences = dataset_loader.parsig_sentences

    def train(self):
        self.set_transition_matrix()
        self.set_emission_matrix()

    def load(self):
        try:
            self.trans_df = pd.read_csv(self.transition_file_name, index_col=0)
            self.emission_df = pd.read_csv(self.emission_file_name, index_col=0)
        except FileNotFoundError:
            print("Trained files not found. Training model first...")
            self.train()

    def get_transition(self, curr_tag, prev_tag):
        count_prev_tag = len([tag for tag in self.all_tags if tag == prev_tag])
        count_prev_tag_curr_tag = 0

        for i in range(len(self.all_tags)-1):
            if self.all_tags[i + 1] == curr_tag and self.all_tags[i] == prev_tag:
                count_prev_tag_curr_tag += 1
        return (count_prev_tag_curr_tag + self.smoothing_const) / (count_prev_tag + self.smoothing_const * self.t)

    def set_transition_matrix(self):
        self.all_tags = [pair[1] for pair in self.train_words_tag]
        self.tags_matrix = np.zeros((len(self.tagset), len(self.tagset)), dtype='float32')

        for i, t1 in enumerate(list(self.tagset)):
            for j, t2 in enumerate(list(self.tagset)):
                self.tags_matrix[i, j] = self.get_transition(t2, t1)

        self.trans_df = pd.DataFrame(self.tags_matrix, columns=list(self.tagset), index=list(self.tagset))
        self.trans_df.to_csv(self.transition_file_name)

    def initialize_counts(self):
        self.word_tag_counts = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)

        for word, tag in self.train_words_tag:
            self.word_tag_counts[word][tag] += 1
            self.tag_counts[tag] += 1

    def get_emission(self, word, tag):
        if word[0] == '<S>' and tag == '<S>':
            return 1.0
        elif word[0] == '<S>' or tag == '<S>':
            return 0.0
        if word[0] not in self.word_tag_counts or tag not in self.word_tag_counts[word[0]]:
            return self.smoothing_const

        word_tag_count = self.word_tag_counts[word[0]][tag]
        tag_count = self.tag_counts[tag]

        return word_tag_count + self.smoothing_const / tag_count + (self.smoothing_const * len(self.vocab))

    def set_emission_matrix(self):
        if not hasattr(self, 'word_tag_counts'):
            self.initialize_counts()
        self.vocab = list(set(word for word, _ in self.train_words_tag))
        self.emission_matrix = np.zeros((len(self.vocab), len(self.tagset)), dtype='float32')
        if self.smoothing_const > 0:
            self.emission_matrix.fill(self.smoothing_const)
        for i, word in enumerate(self.vocab):
            for j, tag in enumerate(self.tagset):
                self.emission_matrix[i, j] = self.get_emission((word, tag), tag)

        self.emission_df = pd.DataFrame(self.emission_matrix, index=self.vocab, columns=self.tagset)
        self.emission_df.to_csv(self.emission_file_name)


    def viterbi(self, words, smoothing=True):
        viterbi = []
        for key, word in enumerate(words):
            if word.endswith('īhā'):
                viterbi.append('ADV')
                continue
            max_prob = -1
            best_tag = None
            for tag in self.tagset:
                if key == 0:
                    transition_p = self.trans_df.loc['<S>', tag]
                else:
                    last_state = viterbi[-1]
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
            viterbi.append(best_tag)
        word_tag = list(zip(words, viterbi))
        return word_tag

    def evaluate(self, tagged_res):
        true_tags = [actual[1] for actual in self.test_run_base]
        pred_tags = [pred[1] for pred in tagged_res]
        wrong_preds = [(p, a) for p, a in zip(tagged_res, self.test_run_base) if p != a]

        metrics = {}
        total_tp = total_fp = total_fn = 0
        total_samples = len(true_tags)

        for tag in self.tagset:
            tp = sum(1 for t, p in zip(true_tags, pred_tags) if t == tag and p == tag)
            fp = sum(1 for t, p in zip(true_tags, pred_tags) if t != tag and p == tag)
            fn = sum(1 for t, p in zip(true_tags, pred_tags) if t == tag and p != tag)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            prec = tp / (tp + fp) if tp + fp > 0 else 0
            rec = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
            support = tp + fn

            metrics[tag] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': support}

        macro = {
            'precision': sum(m['precision'] for m in metrics.values()) / len(self.tagset),
            'recall': sum(m['recall'] for m in metrics.values()) / len(self.tagset),
            'f1': sum(m['f1'] for m in metrics.values()) / len(self.tagset)
        }

        micro_prec = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
        micro_rec = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
        micro = {
            'precision': micro_prec,
            'recall': micro_rec,
            'f1': 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if micro_prec + micro_rec > 0 else 0
        }

        weighted = {
            'precision': sum(m['precision'] * m['support'] for m in metrics.values()) / total_samples,
            'recall': sum(m['recall'] * m['support'] for m in metrics.values()) / total_samples,
            'f1': sum(m['f1'] * m['support'] for m in metrics.values()) / total_samples
        }

        accuracy = (len(tagged_res) - len(wrong_preds)) / len(tagged_res)
        return accuracy, metrics, macro, micro, weighted, wrong_preds

    def print_evaluation(self, tagged_results):
        accuracy, tag_metrics, macro_avg, micro_avg, weighted_avg, wrong_preds = self.evaluate(tagged_results)

        print(f"Accuracy: {accuracy:.3f}")

        print("\nMacro-averaged metrics:")
        for metric, value in macro_avg.items():
            print(f"{metric}: {value:.3f}")

        print("\nWeighted-averaged metrics:")
        for metric, value in weighted_avg.items():
            print(f"{metric}: {value:.3f}")

        print("\nPer-tag metrics:")
        for tag, metrics in tag_metrics.items():
            print(f"{tag}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f} (support={metrics['support']})")


def pos_tagger_evaluation():
    loader = ParsigPOSLoader('Parsig Database.xls')
    tagger = POSTagger(loader)
    tagger.train()
    tagged_results = tagger.viterbi(tagger.test_tagged_words, smoothing=True)
    tagger.print_evaluation(tagged_results)


def run(sentence: str):
    loader = ParsigPOSLoader('Parsig Database.xls', 0)
    tagger = POSTagger(loader, Training = False)
    tagger.load()
    res = []
    for word, tag in tagger.viterbi(("<S> "+ sentence).split(), smoothing=True):
        res.append({'text': word, 'POS': tag})
    return res[1:] # remove the first element due to the added '<S>' tag

if __name__ == '__main__':
    pos_tagger_evaluation()
    print(run("gazīdag abar nihād hēnd be baxt hēnd abar saran, abāz asarīg xwāst hēnd sāg ī garān."))