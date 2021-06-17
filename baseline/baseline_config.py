import torch
config_path = './baseline/baseline_config.py'
bert_vocab_size = 30522


class Baseline_Config():
    output_dir = 'output'
    cuda_idx = 0
    train_device = torch.device('cuda:' + str(cuda_idx))
    batch_size = 16
    dataset = 'IMDB'
    baseline = 'Bert'
    epoch = 30
    save_acc_limit = 0.80

    debug_mode = False

    if_mask_NE = False
    if_replace_NE = False
    if_attach_NE = False
    if_adversial_training = True

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
    adversarial_data_path = r'dataset/IMDB/aclImdb/adversial_instance.std'


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
              'IMDB_Bert_attack_NE_weak': 'output/train_baseline_model/2021-05-19_18:03:20/models/IMDB_Bert_0.91120_05-19-23-35.pt',
              'IMDB_Bert': 'output/train_baseline_model/2021-05-11_21:36:13/models/IMDB_Bert_0.91564_05-11-22-58.pt',
              'IMDB_Bert_adversial_training': 'output/train_baseline_model/2021-06-16_22:56:38/models/IMDB_Bert_0.91304_06-17-02-34.pt',
              'SST2_Bert_attach_NE': 'output/train_baseline_model/2021-05-25_11:01:06/models/SST2_Bert_0.85592_05-25-11-08.pt',
              'SST2_Bert': 'output/train_baseline_model/2021-05-24_22:50:17/models/SST2_Bert_0.87078_05-24-22-59.pt',
              'AGNEWS_Bert': 'output/train_baseline_model/2021-05-25_12:16:00/models/AGNEWS_Bert_0.94250_05-25-14-51.pt',
              'AGNEWS_Bert_attach_NE': 'output/train_baseline_model/2021-05-25_17:30:41/models/AGNEWS_Bert_0.92803_05-26-05-35.pt',
              'IMDB_LSTM': 'output/train_baseline_model/2021-06-09_21:19:25/models/IMDB_LSTM_0.85636_06-09-21-54.pt',
              'IMDB_LSTM_MNE': 'output/train_baseline_model/2021-06-10_21:42:54/models/IMDB_LSTM_0.86404_06-10-22-31.pt',
              'IMDB_LSTM_replace_NE': 'output/train_baseline_model/2021-06-16_18:37:22/models/IMDB_LSTM_0.87340_06-16-19-14.pt',
              'IMDB_LSTM_attach_NE': 'output/train_baseline_model/2021-06-09_22:43:59/models/IMDB_LSTM_0.83464_06-10-04-24.pt',
              'IMDB_LSTM_limit_vocab': 'output/train_baseline_model/2021-06-10_22:03:23/models/IMDB_LSTM_0.85852_06-10-23-12.pt',
              'IMDB_LSTM_limit_vocab_MNE': 'output/train_baseline_model/2021-06-16_17:20:06/models/IMDB_LSTM_0.87360_06-16-18-17.pt',
              'IMDB_LSTM_limit_vocab_replace_NE': 'output/train_baseline_model/2021-06-16_17:18:12/models/IMDB_LSTM_0.87532_06-16-18-14.pt',
              'IMDB_LSTM_limit_vocab_attach_NE': 'output/train_baseline_model/2021-06-11_11:24:58/models/IMDB_LSTM_0.83784_06-11-17-53.pt',
              'IMDB_TextCNN': 'output/train_baseline_model/2021-06-14_22:48:20/models/IMDB_TextCNN_0.86168_06-14-22-53.pt',
              'IMDB_TextCNN_MNE': 'output/train_baseline_model/2021-06-16_17:25:26/models/IMDB_TextCNN_0.84924_06-16-18-03.pt',
              'IMDB_TextCNN_replace_NE': 'output/train_baseline_model/2021-06-16_17:26:22/models/IMDB_TextCNN_0.86164_06-16-18-06.pt',
              'IMDB_TextCNN_attach_NE': 'output/train_baseline_model/2021-06-16_17:27:02/models/IMDB_TextCNN_0.83480_06-16-18-24.pt'}
