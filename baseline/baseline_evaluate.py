from baseline_data import IMDB_Dataset
import torch
from torch.utils.data import DataLoader
from torch import nn
from baseline_model import Baseline_Bert, Baseline_LSTM
from baseline_tools import logging
from baseline_config import Baseline_Config, dataset_config, model_path
from tqdm import tqdm

evaluate_device = torch.device('cuda:0')


def build_dataset(vocab, if_mask_NE, if_replace_NE, if_attach_NE):
    test_dataset_orig = IMDB_Dataset(train_data=False,
                                     vocab=vocab,
                                     if_mask_NE=if_mask_NE,
                                     if_replace_NE=if_replace_NE,
                                     if_attach_NE=if_attach_NE,
                                     debug_mode=False)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Baseline_Config.batch_size,
                           shuffle=False,
                           num_workers=4)
    return test_data


@torch.no_grad()
def evaluate(test_data, baseline_model, criterion):
    baseline_model.eval()
    loss_mean = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(test_data):
        x, y = x.to(evaluate_device), y.to(
            evaluate_device)
        logits = baseline_model(x)
        loss = criterion(logits, y)
        loss_mean += loss.item()

        predicts = logits.argmax(dim=-1)
        correct += predicts.eq(y).float().sum().item()
        total += y.size()[0]

    return loss_mean / len(test_data), correct / total


class BaselineTokenizer():
    def __init__(self, if_mask_NE, if_replace_NE, if_attach_NE):
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          if_mask_NE=if_mask_NE,
                                          if_replace_NE=if_replace_NE,
                                          if_attach_NE=if_attach_NE,
                                          debug_mode=False)
        self.vocab = train_dataset_orig.vocab
        self.tokenizer = train_dataset_orig.tokenizer

    def tokenize(self, sen):
        return self.tokenizer.tokenize(sen)

    def convert_tokens_to_ids(self, word):
        return self.vocab.get_index(word)


if __name__ == '__main__':

    if_mask_NE = False
    if_replace_NE = False
    if_attach_NE = True
    if if_mask_NE:
        model_name = 'IMDB_LSTM_MNE'
    elif if_replace_NE:
        model_name = 'IMDB_LSTM_replace_NE'
    elif if_attach_NE:
        model_name = 'IMDB_LSTM_limit_vocab_attach_NE'
    else:
        model_name = 'IMDB_LSTM_limit_vocab'

    tokenizer = BaselineTokenizer(if_mask_NE, if_replace_NE, if_attach_NE)

    test_data = build_dataset(
        tokenizer.vocab, if_mask_NE, if_replace_NE, if_attach_NE)

    baseline_model = Baseline_LSTM(num_hiddens=128,
                                   num_layers=2,
                                   word_dim=50,
                                   vocab=tokenizer.vocab,
                                   labels_num=2,
                                   using_pretrained=False,
                                   bid=False,
                                   head_tail=False).to(evaluate_device)

    baseline_model.load_state_dict(
        torch.load(model_path[model_name], map_location=evaluate_device))

    criterion = nn.CrossEntropyLoss().to(evaluate_device)

    evaluate_loss, acc = evaluate(test_data, baseline_model, criterion)

    logging(f'acc = {acc}')
