# -*- coding: utf-8 -*-
from .word_stemmer import run as word_stemmer_run
from .p2g import run as p2g_run
from .tokenizer import run as tokenizer_run
from .pos_tagger import run as pos_tagger_run
from .params import Tasks, INVALID_TASKS

TASK2FUNCTION = {
        Tasks.TOKENIZER.value: tokenizer_run,
        Tasks.P2G.value: p2g_run,
        Tasks.LEMMA.value: word_stemmer_run,
        Tasks.POS.value: pos_tagger_run
}


def pipeline(tasks, sentence):
    """
    Pipeline function to run multiple tasks on a sentence.
    
    :param tasks: List of tasks to run on the sentence
    :type tasks: list of Tasks
    :param sentence: Input sentence
    :type sentence: str
    :return: Dictionary containing the output of each task
    """
    unsupported_tasks = [x for x in tasks if x not in Tasks.keys()]
    if unsupported_tasks:
        raise ValueError(INVALID_TASKS.format(unsupported_tasks=', '.join(unsupported_tasks)))
    
    result = {}
    for task in tasks:
        task_output = TASK2FUNCTION[task](sentence)
        result[task] = task_output
    return result
