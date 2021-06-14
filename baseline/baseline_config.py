import torch
config_path = './baseline/baseline_config.py'
bert_vocab_size = 30522


class Baseline_Config():
    output_dir = 'output'
    cuda_idx = 1
    train_device = torch.device('cuda:' + str(cuda_idx))
    batch_size = 64
    dataset = 'IMDB'
    baseline = 'TextCNN'
    epoch = 100
    save_acc_limit = 0.80

    debug_mode = False

    if_mask_NE = False
    if_replace_NE = False

    if_attach_NE = False

    linear_layer_num = 2
    dropout_rate = 0.5
    is_fine_tuning = True

    Bert_lr = 1e-5
    lr = 3e-4
    skip_loss = 0


class IMDBConfig():
    train_data_path = r'./dataset/IMDB/aclImdb/train.std'
    test_data_path = r'./dataset/IMDB/aclImdb/test.std'
    labels_num = 2
    tokenizer_type = 'Bert'
    remove_stop_words = False
    sen_len = 230
    vocab_size = bert_vocab_size


class SST2Config():
    train_data_path = r'./dataset/SST2/train.std'
    test_data_path = r'./dataset/SST2/test.std'
    labels_num = 2
    tokenizer_type = 'Bert'
    remove_stop_words = False
    sen_len = 20
    vocab_size = bert_vocab_size


class AGNEWSConfig():
    train_data_path = r'./dataset/AGNEWS/train.std'
    test_data_path = r'./dataset/AGNEWS/test.std'
    labels_num = 4
    tokenizer_type = 'Bert'
    remove_stop_words = False
    sen_len = 50
    vocab_size = bert_vocab_size


dataset_config = {'IMDB': IMDBConfig,
                  'SST2': SST2Config, 'AGNEWS': AGNEWSConfig}

model_path = {'IMDB_Bert_MNE': 'output/train_baseline_model/2021-05-12_12:57:41/models/IMDB_Bert_0.91328_05-12-14-47.pt',
              'IMDB_Bert_replace_NE': 'output/train_baseline_model/2021-05-13_21:20:48/models/IMDB_Bert_0.91096_05-13-22-37.pt',
              'IMDB_Bert_attach_NE': 'output/train_baseline_model/2021-05-14_21:39:33/models/IMDB_Bert_0.91680_05-15-04-01.pt',
              'IMDB_Bert_attach_NE_inhance': 'output/train_baseline_model/2021-05-18_19:47:01/models/IMDB_Bert_0.91008_05-19-05-14.pt',
              'IMDB_Bert': 'output/train_baseline_model/2021-05-11_21:36:13/models/IMDB_Bert_0.91564_05-11-22-58.pt',
              'SST2_Bert_attach_NE': 'output/train_baseline_model/2021-05-25_11:01:06/models/SST2_Bert_0.85592_05-25-11-08.pt',
              'SST2_Bert': 'output/train_baseline_model/2021-05-24_22:50:17/models/SST2_Bert_0.87078_05-24-22-59.pt',
              'AGNEWS_Bert': 'output/train_baseline_model/2021-05-25_12:16:00/models/AGNEWS_Bert_0.94250_05-25-14-51.pt',
              'AGNEWS_Bert_attach_NE': 'output/train_baseline_model/2021-05-25_17:30:41/models/AGNEWS_Bert_0.92803_05-26-05-35.pt',
              'IMDB_LSTM': 'output/train_baseline_model/2021-06-09_21:19:25/models/IMDB_LSTM_0.84948_06-09-21-54.pt',
              'IMDB_LSTM_attach_NE': 'output/train_baseline_model/2021-06-09_22:43:59/models/IMDB_LSTM_0.82568_06-10-04-24.pt',
              'IMDB_LSTM_limit_vocab': 'output/train_baseline_model/2021-06-10_22:03:23/models/IMDB_LSTM_0.84308_06-10-23-12.pt',
              'IMDB_LSTM_limit_vovab_attach_NE': 'output/train_baseline_model/2021-06-11_11:24:58/models/IMDB_LSTM_0.82988_06-11-17-53.pt',
              'IMDB_LSTM_MNE': 'output/train_baseline_model/2021-06-10_21:42:54/models/IMDB_LSTM_0.84772_06-10-22-31.pt',
              'IMDB_TextCNN': 'output/train_baseline_model/2021-06-14_22:48:20/models/IMDB_TextCNN_0.86168_06-14-22-53.pt'}
