import random
from datetime import datetime
infos = ['INFO', 'WARNING', 'ERROR']
verbose = {0, 2}


def logging(info: str, level: int = 0):
    if level in verbose:
        print('\n\r' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ' ' +
              infos[level] + ' ' + info)


def read_text_data(config_dataset_path, attempt_num):
    datas = []
    labels = []
    with open(config_dataset_path, 'r', encoding='utf-8') as file:
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
