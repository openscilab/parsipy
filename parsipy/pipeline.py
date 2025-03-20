# -*- coding: utf-8 -*-
"""pipeline."""
from .word_stemmer import run as word_stemmer_run
from .p2t import run as p2t_run
from .tokenizer import run as tokenizer_run
from .pos_tagger import run as pos_tagger_run
from .params import Task, INVALID_TASKS

TASK2FUNCTION = {
    Task.TOKENIZER.value: tokenizer_run,
    Task.P2T.value: p2t_run,
    Task.LEMMA.value: word_stemmer_run,
    Task.POS.value: pos_tagger_run
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
    unsupported_tasks = [x for x in tasks if x not in Task]
    if unsupported_tasks:
        raise ValueError(INVALID_TASKS.format(unsupported_tasks=', '.join(unsupported_tasks)))

    result = {}
    for task in tasks:
        task_output = TASK2FUNCTION[task.value](sentence)
        result[task.value] = task_output
    return result
