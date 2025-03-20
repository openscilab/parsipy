# -*- coding: utf-8 -*-
"""parameters and constants."""
from enum import Enum

PARSIPY_VERSION = "0.1"

INVALID_TASKS = "Sorry, the following tasks are not supported yet: {unsupported_tasks}"

POS_MAPPING = {
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


class POSTaggerModel(Enum):
    """POSTagger models."""

    RULE_BASED = "rule_based"


class Task(Enum):
    """Tasks supported by Parsipy."""

    LEMMA = "lemma"
    POS = "POS"
    P2T = "P2T"
    TOKENIZER = "tokenizer"
