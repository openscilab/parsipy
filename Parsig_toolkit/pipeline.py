import word_stemmer
import p2g
import tokenizer

provided_tasks = ['P2G', 'lemma', 'POS', 'tokenizer']


def run(tasks: str, sentence: str):
    tasks_list = [task.strip() for task in tasks.split(',')]
    invalid_tasks = [task for task in tasks_list if task not in provided_tasks]
    if invalid_tasks:
        raise ValueError(f"Sorry, the following tasks are not provided yet: {', '.join(invalid_tasks)}")

    final_output = tokenizer.run(sentence)

    task_to_function = {
        'P2G': p2g.run,
        'lemma': word_stemmer.run
    }

    for task in tasks_list:
        if task in task_to_function:
            task_output = task_to_function[task](sentence)
            final_output = [current_output | task_output for current_output, task_output in
                            zip(final_output, task_output)]
    return final_output


if __name__ == '__main__':
    print(run(tasks='P2G, lemma, NER', sentence='ruwān ī xwēš andar ayād dār nām ī xwēš rāy xwēš-kārīh'))
