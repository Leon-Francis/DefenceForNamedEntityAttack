from datetime import datetime
import os
from spellchecker import SpellChecker
import torch
import csv
import numpy as np
import random

random.seed(667)

spell = SpellChecker()

infos = ['INFO', 'WARNING', 'ERROR']
verbose = {0, 2}


def logging(info: str, level: int = 0):
    if level in verbose:
        print('\n\r' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ' ' +
              infos[level] + ' ' + info)


def get_time() -> str:
    return str(datetime.now().strftime("%m-%d-%H:%M"))


def make_dir_if_not_exist(path):
    exist = os.path.exists(path)
    if not exist:
        os.makedirs(path)


def parse_bool(x):
    if x == 'yes':
        return True
    if x == 'no':
        return False
    return None


def read_text_test_data(config_dataset_path, attempt_num):
    datas = []
    labels = []
    with open(config_dataset_path, 'r',
              encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            datas.append(line[:-1])
            labels.append(int(line[-1]))
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(datas)
    random.seed(randnum)
    random.shuffle(labels)
    datas = datas[:attempt_num]
    labels = labels[:attempt_num]
    logging(f'loading data {len(datas)}')
    return datas, labels


def read_text_train_data(config_dataset_path):
    datas = []
    labels = []
    with open(config_dataset_path, 'r',
              encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            datas.append(line[:-1])
            labels.append(int(line[-1]))
    logging(f'loading data {len(datas)}')
    return datas, labels


def read_IMDB_origin_data(data_path):
    path_list = []
    logging(f'start loading data from {data_path}')
    dirs = os.listdir(data_path)
    for dir in dirs:
        if dir == 'pos' or dir == 'neg':
            file_list = os.listdir(os.path.join(data_path, dir))
            file_list = map(lambda x: os.path.join(data_path, dir, x),
                            file_list)
            path_list += list(file_list)
    datas = []
    labels = []
    for p in path_list:
        label = 0 if 'neg' in p else 1
        with open(p, 'r', encoding='utf-8') as file:
            datas.append(file.readline())
            labels.append(label)

    return datas, labels


def read_AGNEWS_origin_data(data_path):
    datas = []
    labels = []
    with open(data_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            labels.append(int(line[0]) - 1)
            datas.append(line[1] + '. ' + line[2])
    return datas, labels


def read_YAHOO_origin_data(data_path):
    datas = []
    labels = []
    dirs = os.listdir(data_path)
    for idx, dir in enumerate(dirs):
        path = os.path.join(data_path, dir)
        if os.path.isfile(path):
            continue
        for file in os.listdir(path):
            tpath = os.path.join(path, file)
            with open(tpath, 'r', encoding='utf-8', newline='') as t:
                datas.append(t.readline())
                labels.append(idx)
    return datas, labels


def read_YAHOO_CSV(data_path):
    datas = []
    labels = []
    with open(data_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            labels.append(int(line[0]) - 1)
            datas.append(''.join(line[1:]))

    return datas, labels


def read_standard_data(path):
    data = []
    labels = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            data.append(line[:-1])
            labels.append(int(line[-1]))
    logging(f'loading data {len(data)} from {path}')
    return data, labels


def write_standard_data(datas, labels, path, mod='w'):
    assert len(datas) == len(labels)
    num = len(labels)
    logging(f'writing standard data {num} to {path}')
    with open(path, mod, newline='', encoding='utf-8') as file:
        for i in range(num):
            file.write(datas[i] + str(labels[i]) + '\n')


def str2tokens(sentence: str, tokenizer):
    return tokenizer.tokenize(sentence)


def tokens2seq(tokens, maxlen: int, tokenizer) -> torch.Tensor:
    pad_word = 0
    x = [pad_word for _ in range(maxlen)]
    temp = tokens[:maxlen]
    for idx, word in enumerate(temp):
        x[idx] = tokenizer.convert_tokens_to_ids(word)
    return torch.tensor(x)


def str2seq(sentence: str, maxlen: int, tokenizer) -> torch.Tensor:
    tokens = str2tokens(sentence, tokenizer)
    return tokens2seq(tokens, maxlen, tokenizer)


def read_fool_log(path):
    with open(path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        temp = []
        for idx, line in enumerate(reader):
            if (idx + 1) % 2 == 1:
                temp.append(line)

    count = len(temp)
    temp = np.array(temp, dtype='float32')
    temp = np.sum(temp, axis=0)
    temp /= count
    temp = temp.tolist()
    return (count, temp[-2], temp[-1])


def get_random(s: int, e: int, weights=None):
    if weights is not None:
        return random.choices([i for i in range(s, e + 1)], weights)[0]
    return random.randint(s, e)


def np_softmax(x: list):
    x = np.exp(x)
    return x / np.sum(x, axis=0)
