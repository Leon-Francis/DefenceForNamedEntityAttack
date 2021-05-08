from transformers import BertTokenizer
import spacy
from torch.utils.data import Dataset
from data_tools import logging
from baseline_config import IMDBConfig
nlp = spacy.load('en_core_web_sm')


class IMDB_Dataset(Dataset):
    def __init__(self, train_data=True, if_mask_NE=False, debug_mode=False):
        super(IMDB_Dataset, self).__init__()
        if train_data:
            self.path = IMDBConfig.train_data_path
        else:
            self.path = IMDBConfig.test_data_path
        self.datas, self.classification_label = self.read_standard_data(
            self.path, debug_mode)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sen_len = IMDBConfig.sen_len
        self.data_tokens = []
        self.data_idx = []
        self.data2tokens(if_mask_NE)
        self.token2idx()
        self.transfor()

    def read_standard_data(self, path, debug_mode=False):
        data = []
        labels = []
        if debug_mode:
            i = 1000
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

    def data2tokens(self, if_mask_NE):
        logging(f'{self.path} in data2tokens')
        if if_mask_NE:
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
                tokens = self.tokenizer.tokenize(masked_string)[:self.sen_len]
                self.data_tokens.append(tokens)
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