# -*- coding: utf-8 -*-
from .params import PARSIPY_VERSION, POSTaggerModel
from .word_stemmer import find_root
from .tokenizer import preprocess, cleaning
from .pos_tagger import POSTagger
from .p2g import oov_translit, get_transliteration
from .pipeline import pipeline

__version__ = PARSIPY_VERSION