import __init__paths
import torch
from transformers import BertTokenizer
from baseline_model import Baseline_Bert
from config import BertConfig, config_dataset, model_path, IMDBConfig
from baseline_config import dataset_config
from tools import read_text_data, logging
from random import choice
import json
import spacy
import numpy as np
nlp = spacy.load('en_core_web_sm')
attempt_num = 100
N = 100
config_device = torch.device('cuda:0')
attach_NE = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
datas, labels = read_text_data(dataset_config[config_dataset].test_data_path,
                               attempt_num)
baseline_model = Baseline_Bert(
    label_num=dataset_config[config_dataset].labels_num,
    linear_layer_num=BertConfig.linear_layer_num[config_dataset],
    dropout_rate=BertConfig.dropout_rate[config_dataset],
    is_fine_tuning=BertConfig.is_fine_tuning[config_dataset]).to(config_device)

baseline_model.load_state_dict(
    torch.load(model_path['IMDB_Bert_attach_NE_inhance'],
               map_location=config_device))

baseline_model.eval()

f_0 = open('pwws/NE_dict/imdb_adv_0.json', 'r')
content_0 = f_0.read()
imdb_0 = json.loads(content_0)
f_0.close()
f_1 = open('pwws/NE_dict/imdb_adv_1.json', 'r')
content_1 = f_1.read()
imdb_1 = json.loads(content_1)
f_1.close()
imdb_attach_NE = [imdb_0, imdb_1]

sen_len = IMDBConfig.padding_maxlen

PSS_Score = 0
PSD_Score = 0
PSR_Score = 0

for sen_idx, sen in enumerate(datas):
    ori_data_tokens = tokenizer.tokenize(sen)[:sen_len]
    preturb_data_tokens = []
    doc = nlp(sen)

    for _ in range(N):
        tokens = []
        for token in doc:
            string = str(token)
            tokens.append(string)
        for ent in doc.ents:
            tokens[ent.start] = choice(
                imdb_attach_NE[labels[sen_idx]][ent.label_])
            for idx in range(ent.start + 1, ent.end):
                tokens[idx] = ''
        attach_NE_string = ' '.join(tokens)
        tokens = tokenizer.tokenize(attach_NE_string)[:sen_len]
        preturb_data_tokens.append(tokens)

    ori_data_idx = tokenizer.convert_tokens_to_ids(
        ori_data_tokens)  # batch = 1
    perturb_data_idx = []
    for tokens in preturb_data_tokens:
        perturb_data_idx.append(
            tokenizer.convert_tokens_to_ids(tokens))  # batch = N

    if len(ori_data_idx) < sen_len:
        ori_data_idx += [0] * (sen_len - len(ori_data_idx))

    for i in range(N):
        if len(perturb_data_idx[i]) < sen_len:
            perturb_data_idx[i] += [0] * (sen_len - len(perturb_data_idx[i]))

    ori_data_idx = torch.tensor(ori_data_idx).to(config_device)
    perturb_data_idx = torch.tensor(perturb_data_idx).to(config_device)

    ori_prob = baseline_model.predict_prob(
        ori_data_idx,
        torch.zeros(1, dtype=torch.long).to(config_device))

    perturb_prob = baseline_model.predict_prob(
        perturb_data_idx,
        torch.zeros(N, dtype=torch.long).to(config_device))

    for i in range(N):
        PSS_Score += abs(perturb_prob[i] - ori_prob[0])
    PSS_Score /= N
    PSD_Score += np.std(perturb_prob)
    PSR_Score += max(perturb_prob) - min(perturb_prob)

PSS_Score /= 100
PSD_Score /= 100
PSR_Score /= 100
logging(f'PSS_Score ={PSS_Score}')
logging(f'PSD_Score ={PSD_Score}')
logging(f'PSR_Score ={PSR_Score}')