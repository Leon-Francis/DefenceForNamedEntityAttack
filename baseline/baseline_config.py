import torch
config_path = './baseline/baseline_config.py'
bert_vocab_size = 30522


class Baseline_Config:
    output_dir = 'output'
    cuda_idx = 0
    train_device = torch.device('cuda:' + str(cuda_idx))
    batch_size = 8
    dataset = 'IMDB'
    baseline = 'Bert'
    epoch = 15
    save_acc_limit = 0.85

    debug_mode = False

    if_mask_NE = False
    if_replace_NE = False
    if_attach_NE = True

    linear_layer_num = 1
    dropout_rate = 0.5
    is_fine_tuning = True

    Bert_lr = 1e-5
    lr = 1e-3
    skip_loss = 0.16


class IMDBConfig():
    train_data_path = r'./dataset/IMDB/aclImdb/train.std'
    test_data_path = r'./dataset/IMDB/aclImdb/test.std'
    labels_num = 2
    tokenizer_type = 'Bert'
    remove_stop_words = False
    sen_len = 230
    vocab_size = bert_vocab_size


dataset_config = {'IMDB': IMDBConfig}

model_path = {'IMDB_Bert_MNE': 'output/train_baseline_model/2021-05-12_12:57:41/models/IMDB_Bert_0.91328_05-12-14-47.pt',
              'IMDB_Bert_replace_NE': 'output/train_baseline_model/2021-05-13_21:20:48/models/IMDB_Bert_0.91096_05-13-22-37.pt',
              'IMDB_Bert_attach_NE': 'output/train_baseline_model/2021-05-14_21:39:33/models/IMDB_Bert_0.91680_05-15-04-01.pt',
              'IMDB_Bert': 'output/train_baseline_model/2021-05-11_21:36:13/models/IMDB_Bert_0.91564_05-11-22-58.pt'}
