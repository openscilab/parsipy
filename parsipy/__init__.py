# -*- coding: utf-8 -*-
from .params import PARSIPY_VERSION
from .word_stemmer import find_root
from .tokenizer import preprocess, cleaning
from .pos_tagger import POSTaggerRuleBased
from .p2g import oov_translit, get_transliteration

__version__ = PARSIPY_VERSION