from transformers import BertTokenizer
import spacy
import torch
from torch.utils.data import Dataset
from baseline_tools import logging, load_pkl_obj, save_pkl_obj
from baseline_config import Baseline_Config, IMDBConfig, SST2Config, AGNEWSConfig
from random import choice
import json
import re
import numpy as np
import os
nlp = spacy.load('en_core_web_sm')


class IMDB_Dataset(Dataset):
    def __init__(self,
                 train_data=True,
                 vocab=None,
                 if_mask_NE=False,
                 if_replace_NE=False,
                 if_attach_NE=False,
                 debug_mode=False):
        super(IMDB_Dataset, self).__init__()
        self.train_model = train_data
        if train_data:
            self.path = IMDBConfig.train_data_path
        else:
            self.path = IMDBConfig.test_data_path
        self.datas, self.classification_label = self.read_standard_data(
            self.path, debug_mode)
        if Baseline_Config.baseline == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = Baseline_Tokenizer()
        self.sen_len = IMDBConfig.sen_len
        self.data_tokens = []
        self.data_idx = []
        self.replace_dict = {
            'PERSON': 'name',
            'NORP': 'group',
            'FAC': 'building',
            'ORG': 'company',
            'GPE': 'country',
            'LOC': 'location',
            'PRODUCT': 'object',
            'EVENT': 'event',
            'WORK_OF_ART': 'book',
            'LAW': 'law',
            'LANGUAGE': 'language',
            'DATE': 'date',
            'TIME': 'time',
            'PERCENT': 'percentage',
            'MONEY': 'money',
            'QUANTITY': 'quantity',
            'ORDINAL': 'ordinal',
            'CARDINAL': 'number'
        }
        f_0 = open('pwws/NE_dict/imdb_adv_0.json', 'r')
        content_0 = f_0.read()
        imdb_0 = json.loads(content_0)
        f_0.close()
        f_1 = open('pwws/NE_dict/imdb_adv_1.json', 'r')
        content_1 = f_1.read()
        imdb_1 = json.loads(content_1)
        f_1.close()
        self.imdb_attach_NE = [imdb_0, imdb_1]
        self.data2tokens(if_mask_NE, if_replace_NE, if_attach_NE)
        if Baseline_Config.baseline == 'Bert':
            self.vocab = None
        else:
            if not vocab:
                self.vocab = Baseline_Vocab(self.data_tokens)
            else:
                self.vocab = vocab
        self.token2idx()
        self.transfor()

    def read_standard_data(self, path, debug_mode=False):
        data = []
        labels = []
        if debug_mode:
            i = 100
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    i -= 1
                    line = line.strip('\n')
                    data.append(line[:-1])
                    labels.append(int(line[-1]))
                    if i == 0:
                        break
            logging(f'loading data {len(data)} from {path}')
            return data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                data.append(line[:-1])
                labels.append(int(line[-1]))
        logging(f'loading data {len(data)} from {path}')
        return data, labels

    def data2tokens(self, if_mask_NE, if_replace_NE, if_attach_NE):
        logging(f'{self.path} in data2tokens')
        if self.train_model and if_mask_NE:
            for sen in self.datas:
                doc = nlp(sen)
                tokens = []
                for token in doc:
                    string = str(token)
                    tokens.append(string)
                for ent in doc.ents:
                    for idx in range(ent.start, ent.end):
                        tokens[idx] = '[MASK]'
                masked_NE_string = ' '.join(tokens)
                tokens = self.tokenizer.tokenize(
                    masked_NE_string)[:self.sen_len]
                self.data_tokens.append(tokens)
        elif self.train_model and if_replace_NE:
            for sen in self.datas:
                doc = nlp(sen)
                tokens = []
                for token in doc:
                    string = str(token)
                    tokens.append(string)
                for ent in doc.ents:
                    tokens[ent.start] = self.replace_dict[ent.label_]
                    for idx in range(ent.start + 1, ent.end):
                        tokens[idx] = ''
                replaced_NE_string = ' '.join(tokens)
                tokens = self.tokenizer.tokenize(
                    replaced_NE_string)[:self.sen_len]
                self.data_tokens.append(tokens)
        elif self.train_model and if_attach_NE:
            temp_label_list = []
            NE_samples = 0
            NE_nums = 0
            for sen_idx, sen in enumerate(self.datas):
                tokens = self.tokenizer.tokenize(sen)[:self.sen_len]
                self.data_tokens.append(tokens)
                temp_label_list.append(self.classification_label[sen_idx])
                doc = nlp(sen)
                if (len(doc.ents) > 0):
                    NE_samples += 1
                    NE_nums += len(doc.ents)
                for _ in range(len(doc.ents)):
                    tokens = []
                    for token in doc:
                        string = str(token)
                        tokens.append(string)
                    for ent in doc.ents:
                        tokens[ent.start] = choice(self.imdb_attach_NE[
                            self.classification_label[sen_idx]][ent.label_])
                        for idx in range(ent.start + 1, ent.end):
                            tokens[idx] = ''
                    attach_NE_string = ' '.join(tokens)
                    tokens = self.tokenizer.tokenize(
                        attach_NE_string)[:self.sen_len]
                    self.data_tokens.append(tokens)
                    temp_label_list.append(self.classification_label[sen_idx])
            self.classification_label = temp_label_list
            logging(f'NE samples = {NE_samples}\nNE nums = {NE_nums}')
        else:
            for sen in self.datas:
                tokens = self.tokenizer.tokenize(sen)[:self.sen_len]
                self.data_tokens.append(tokens)

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        if not self.vocab:
            for tokens in self.data_tokens:
                self.data_idx.append(
                    self.tokenizer.convert_tokens_to_ids(tokens))
        else:
            for tokens in self.data_tokens:
                self.data_idx.append(
                    [self.vocab.get_index(token) for token in tokens])

        for i in range(len(self.data_idx)):
            if len(self.data_idx[i]) < self.sen_len:
                self.data_idx[i] += [0
                                     ] * (self.sen_len - len(self.data_idx[i]))

    def transfor(self):
        self.data_idx = torch.tensor(self.data_idx)
        self.classification_label = torch.tensor(self.classification_label)

    def __getitem__(self, item):
        return self.data_idx[item], self.classification_label[item]

    def __len__(self):
        return len(self.data_idx)


class SST2_Dataset(Dataset):
    def __init__(self, train_data=True, if_attach_NE=False, debug_mode=False):
        super(SST2_Dataset, self).__init__()
        self.train_model = train_data
        if train_data:
            self.path = SST2Config.train_data_path
        else:
            self.path = SST2Config.test_data_path
        self.datas, self.classification_label = self.read_standard_data(
            self.path, debug_mode)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sen_len = SST2Config.sen_len
        self.data_tokens = []
        self.data_idx = []

        f_0 = open('pwws/NE_dict/SST2_adv_0.json', 'r')
        content_0 = f_0.read()
        sst2_0 = json.loads(content_0)
        f_0.close()
        f_1 = open('pwws/NE_dict/SST2_adv_1.json', 'r')
        content_1 = f_1.read()
        sst2_1 = json.loads(content_1)
        f_1.close()
        self.sst2_attach_NE = [sst2_0, sst2_1]

        self.data2tokens(if_attach_NE)
        self.token2idx()
        self.transfor()

    def read_standard_data(self, path, debug_mode=False):
        data = []
        labels = []
        if debug_mode:
            i = 100
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    i -= 1
                    line = line.strip('\n')
                    data.append(line[:-1])
                    labels.append(int(line[-1]))
                    if i == 0:
                        break
            logging(f'loading data {len(data)} from {path}')
            return data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                data.append(line[:-1])
                labels.append(int(line[-1]))
        logging(f'loading data {len(data)} from {path}')
        return data, labels

    def data2tokens(self, if_attach_NE):
        logging(f'{self.path} in data2tokens')
        if self.train_model and if_attach_NE:
            temp_label_list = []
            NE_samples = 0
            NE_nums = 0
            for sen_idx, sen in enumerate(self.datas):
                tokens = self.tokenizer.tokenize(sen)[:self.sen_len]
                self.data_tokens.append(tokens)
                temp_label_list.append(self.classification_label[sen_idx])
                doc = nlp(sen)
                if (len(doc.ents) > 0):
                    NE_samples += 1
                    NE_nums += len(doc.ents)
                for _ in range(len(doc.ents)):
                    tokens = []
                    for token in doc:
                        string = str(token)
                        tokens.append(string)
                    for ent in doc.ents:
                        tokens[ent.start] = choice(self.sst2_attach_NE[
                            self.classification_label[sen_idx]][ent.label_])
                        for idx in range(ent.start + 1, ent.end):
                            tokens[idx] = ''
                    attach_NE_string = ' '.join(tokens)
                    tokens = self.tokenizer.tokenize(
                        attach_NE_string)[:self.sen_len]
                    self.data_tokens.append(tokens)
                    temp_label_list.append(self.classification_label[sen_idx])
            self.classification_label = temp_label_list
            logging(f'NE samples = {NE_samples}\nNE nums = {NE_nums}')
        else:
            for sen in self.datas:
                tokens = self.tokenizer.tokenize(sen)[:self.sen_len]
                self.data_tokens.append(tokens)

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        for tokens in self.data_tokens:
            self.data_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))

        for i in range(len(self.data_idx)):
            if len(self.data_idx[i]) < self.sen_len:
                self.data_idx[i] += [0
                                     ] * (self.sen_len - len(self.data_idx[i]))

    def transfor(self):
        self.data_idx = torch.tensor(self.data_idx)
        self.classification_label = torch.tensor(self.classification_label)

    def __getitem__(self, item):
        return self.data_idx[item], self.classification_label[item]

    def __len__(self):
        return len(self.data_idx)


class AGNEWS_Dataset(Dataset):
    def __init__(self, train_data=True, if_attach_NE=False, debug_mode=False):
        super(AGNEWS_Dataset, self).__init__()
        self.train_model = train_data
        if train_data:
            self.path = AGNEWSConfig.train_data_path
        else:
            self.path = AGNEWSConfig.test_data_path
        self.datas, self.classification_label = self.read_standard_data(
            self.path, debug_mode)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sen_len = AGNEWSConfig.sen_len
        self.data_tokens = []
        self.data_idx = []

        f_0 = open('pwws/NE_dict/AGNEWS_adv_0.json', 'r')
        content_0 = f_0.read()
        agnews_0 = json.loads(content_0)
        f_0.close()
        f_1 = open('pwws/NE_dict/AGNEWS_adv_1.json', 'r')
        content_1 = f_1.read()
        agnews_1 = json.loads(content_1)
        f_1.close()
        f_2 = open('pwws/NE_dict/AGNEWS_adv_2.json', 'r')
        content_2 = f_2.read()
        agnews_2 = json.loads(content_2)
        f_2.close()
        f_3 = open('pwws/NE_dict/AGNEWS_adv_3.json', 'r')
        content_3 = f_3.read()
        agnews_3 = json.loads(content_3)
        f_3.close()

        self.agnews_attach_NE = [agnews_0, agnews_1, agnews_2, agnews_3]

        self.data2tokens(if_attach_NE)
        self.token2idx()
        self.transfor()

    def read_standard_data(self, path, debug_mode=False):
        data = []
        labels = []
        if debug_mode:
            i = 100
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    i -= 1
                    line = line.strip('\n')
                    data.append(line[:-1])
                    labels.append(int(line[-1]))
                    if i == 0:
                        break
            logging(f'loading data {len(data)} from {path}')
            return data, labels
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                data.append(line[:-1])
                labels.append(int(line[-1]))
        logging(f'loading data {len(data)} from {path}')
        return data, labels

    def data2tokens(self, if_attach_NE):
        logging(f'{self.path} in data2tokens')
        if self.train_model and if_attach_NE:
            temp_label_list = []
            NE_samples = 0
            NE_nums = 0
            for sen_idx, sen in enumerate(self.datas):
                tokens = self.tokenizer.tokenize(sen)[:self.sen_len]
                self.data_tokens.append(tokens)
                temp_label_list.append(self.classification_label[sen_idx])
                doc = nlp(sen)
                if (len(doc.ents) > 0):
                    NE_samples += 1
                    NE_nums += len(doc.ents)
                for _ in range(len(doc.ents)):
                    tokens = []
                    for token in doc:
                        string = str(token)
                        tokens.append(string)
                    for ent in doc.ents:
                        tokens[ent.start] = choice(self.agnews_attach_NE[
                            self.classification_label[sen_idx]][ent.label_])
                        for idx in range(ent.start + 1, ent.end):
                            tokens[idx] = ''
                    attach_NE_string = ' '.join(tokens)
                    tokens = self.tokenizer.tokenize(
                        attach_NE_string)[:self.sen_len]
                    self.data_tokens.append(tokens)
                    temp_label_list.append(self.classification_label[sen_idx])
            self.classification_label = temp_label_list
            logging(f'NE samples = {NE_samples}\nNE nums = {NE_nums}')
        else:
            for sen in self.datas:
                tokens = self.tokenizer.tokenize(sen)[:self.sen_len]
                self.data_tokens.append(tokens)

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        for tokens in self.data_tokens:
            self.data_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))

        for i in range(len(self.data_idx)):
            if len(self.data_idx[i]) < self.sen_len:
                self.data_idx[i] += [0
                                     ] * (self.sen_len - len(self.data_idx[i]))

    def transfor(self):
        self.data_idx = torch.tensor(self.data_idx)
        self.classification_label = torch.tensor(self.classification_label)

    def __getitem__(self, item):
        return self.data_idx[item], self.classification_label[item]

    def __len__(self):
        return len(self.data_idx)


class Baseline_Vocab():
    def __init__(self,
                 origin_data_tokens,
                 vocab_limit_size=80000,
                 is_special=False,
                 is_using_pretrained=True,
                 word_vec_file_path=r'./static/glove.6B.100d.txt'):
        assert len(origin_data_tokens) > 0
        self.file_path = word_vec_file_path
        self.word_dim = int(re.findall("\d+d", word_vec_file_path)[0][:-1])
        self.word_dict = {}
        self.word_count = {}
        self.vectors = None
        self.num = 0
        self.data_tokens = []
        self.words_vocab = []
        self.is_special = is_special  # enable <cls> and <sep>
        self.special_word_pad = ('[PAD]', 0)
        self.special_word_unk = ('[UNK]', 1)
        self.special_word_cls = ('[CLS]', 2)
        self.special_word_sep = ('[SEP]', 3)
        self.data_tokens = origin_data_tokens
        self.__build_words_index()
        self.__limit_dict_size(vocab_limit_size)
        if is_using_pretrained:
            logging(f'building word vectors from {self.file_path}')
            self.__read_pretrained_word_vecs()
        logging(f'word vectors has been built! dict size is {self.num}')

    def __build_words_index(self):
        for sen in self.data_tokens:
            for word in sen:
                if word not in self.word_dict:
                    self.word_dict[word] = self.num
                    self.word_count[word] = 1
                    self.num += 1
                else:
                    self.word_count[word] += 1

    def __limit_dict_size(self, vocab_limit_size):
        limit = vocab_limit_size
        word_count_temp = sorted(self.word_count.items(),
                                 key=lambda x: x[1],
                                 reverse=True)
        count = 2
        self.words_vocab.append(self.special_word_pad[0])
        self.words_vocab.append(self.special_word_unk[0])
        self.word_count[self.special_word_pad[0]] = int(1e9)
        self.word_count[self.special_word_unk[0]] = int(1e9)
        if self.is_special:
            self.words_vocab += [
                self.special_word_cls[0], self.special_word_sep[0]
            ]
            self.word_count[self.special_word_cls[0]] = int(1e9)
            self.word_count[self.special_word_sep[0]] = int(1e9)
            count += 2
        temp = {}
        for x, y in word_count_temp:
            if count > limit:
                break
            temp[x] = count
            self.words_vocab.append(x)
            count += 1
        self.word_dict = temp
        self.word_dict[self.special_word_pad[0]] = 0
        self.word_dict[self.special_word_unk[0]] = 1
        if self.is_special:
            self.word_dict[self.special_word_cls[0]] = 2
            self.word_dict[self.special_word_sep[0]] = 3
        self.num = count
        assert self.num == len(self.word_dict) == len(self.words_vocab)
        self.vectors = np.ndarray([self.num, self.word_dim], dtype='float32')

    def __read_pretrained_word_vecs(self):
        num = 2
        word_dict = {}
        word_dict[self.special_word_pad[0]] = 0
        word_dict[self.special_word_unk[0]] = 1  # unknown word

        temp = self.file_path + '.pkl'
        if os.path.exists(temp):
            word_dict, vectors = load_pkl_obj(temp)
        else:
            if self.is_special:
                word_dict[self.special_word_cls[0]] = 2
                word_dict[self.special_word_sep[0]] = 3
                num += 2
            with open(self.file_path, 'r', encoding='utf-8') as file:
                file = file.readlines()
                vectors = np.ndarray([len(file) + num, self.word_dim],
                                     dtype='float32')
                vectors[0] = np.random.normal(0.0, 0.3, [self.word_dim])  # pad
                vectors[1] = np.random.normal(0.0, 0.3, [self.word_dim])  # unk
                if self.is_special:
                    vectors[2] = np.random.normal(0.0, 0.3, [self.word_dim])
                    vectors[3] = np.random.normal(0.0, 0.3, [self.word_dim])
                for line in file:
                    line = line.split()
                    word_dict[line[0]] = num
                    vectors[num] = np.asarray(line[-self.word_dim:],
                                              dtype='float32')
                    num += 1

            save_pkl_obj((word_dict, vectors), temp)

        for word, idx in self.word_dict.items():
            if word in word_dict:
                key = word_dict[word]
                self.vectors[idx] = vectors[key]
            else:
                self.vectors[idx] = vectors[1]

    def __len__(self):
        return self.num

    def get_word_count(self, word):
        # word could be int or str
        if isinstance(word, int):
            word = self.get_word(word)
        if word not in self.word_count:
            return 0  # OOV
        return self.word_count[word]

    def get_index(self, word: str):
        if word not in self.word_dict:
            return 1  # unknown word
        return self.word_dict[word]

    def get_word(self, index: int):
        return self.words_vocab[index]

    def get_vec(self, index: int):
        assert self.vectors is not None
        return self.vectors[index]


class Baseline_Tokenizer():
    def __init__(self):
        pass

    def pre_process(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"<br />", "", text)
        text = re.sub(r'(\W)(?=\1)', '', text)
        text = re.sub(r"([.!?,])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
        return text.strip()

    def normal_token(self, text: str):
        return [tok for tok in text.split() if not tok.isspace()]

    def tokenize(self, text: str):
        text = self.pre_process(text)
        words = self.normal_token(text)
        return words


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = 'Bill Paxton stars in and directs this highly original film. Having watched the first time I was by how good it was. The reviews I had heard were OK . As a result I was expecting an average thriller at most .However because of Paxtons excellent directing and acting the film is well worth watching , especially if you are a horror film fanatic.The film is also helped by the plot twists which keep coming until the closing credits . The films strongest point is the storyline which I have to say is highly original and is like I have ever seen before. Well done also to the 2 young leads which perfectly convey the emotions if these confused boys. I give this film 9/10 and I highly recommend that everyone catches it.'
    tokens = tokenizer.tokenize(text)
    print(tokens)
    seqs = tokenizer.convert_tokens_to_ids(tokens)
    print(seqs)
    doc = nlp(text)
    tokens = []
    for token in doc:
        string = str(token)
        tokens.append(string)
    for ent in doc.ents:
        print(
            ent.text,
            ent.label_,
        )
        for idx in range(ent.start, ent.end):
            tokens[idx] = '[MASK]'
    masked_string = ' '.join(tokens)
    print(masked_string)
    tokens = tokenizer.tokenize(masked_string)
    print(tokens)
    seqs = tokenizer.convert_tokens_to_ids(tokens)
    print(seqs)
