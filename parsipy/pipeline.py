# -*- coding: utf-8 -*-
from .word_stemmer import run as word_stemmer_run
from .p2g import run as p2g_run
from .tokenizer import run as tokenizer_run
from .pos_tagger import run as pos_tagger_run

PROVIDED_TASKS = ['P2G', 'lemma', 'POS', 'tokenizer']


def pipeline(tasks, sentence):
    tasks_list = [task.strip() for task in tasks.split(',')]
    invalid_tasks = [task for task in tasks_list if task not in PROVIDED_TASKS]
    if invalid_tasks:
        raise ValueError("Sorry, the following tasks are not provided yet: {invalid_tasks}".format(invalid_tasks=', '.join(invalid_tasks)))
    final_output = dict()
    final_output["tokenizer"] = tokenizer_run(sentence)

    task_to_function = {
        'P2G': p2g_run,
        'lemma': word_stemmer_run,
        'POS': pos_tagger_run
    }

    for task in tasks_list:
        if task in task_to_function:
            task_output = task_to_function[task](sentence)
            final_output[task] = task_output
    return final_output
