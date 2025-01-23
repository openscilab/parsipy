# -*- coding: utf-8 -*-
def cleaning(text: str):
    return text.replace('*', '').replace('(w)', '').replace('[', '').replace(']', '').replace('<', '').replace('>',
                                                                                                               '')


def preprocess(sentence: str):
    sentence = sentence.replace('-', ' ')
    return cleaning(sentence)


def run(sentence: str):
    sentence = preprocess(sentence)
    my_list = []
    for id, word in enumerate(sentence.split()):
        data = {'id': id, 'text': word}
        my_list.append(data)
    return my_list


if __name__ == '__main__':
    print(run('ruwān ī xwēš andar ayād dār nām ī xwēš rāy xwēš-kārīh '))
