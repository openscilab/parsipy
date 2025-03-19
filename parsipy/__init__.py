# -*- coding: utf-8 -*-
"""parsipy module."""
from .params import Task, PARSIPY_VERSION, POSTaggerModel
from .word_stemmer import find_root
from .tokenizer import preprocess, cleaning
from .pos_tagger import POSTagger
from .p2t import oov_transliterate, get_transliteration
from .pipeline import pipeline

__version__ = PARSIPY_VERSION
