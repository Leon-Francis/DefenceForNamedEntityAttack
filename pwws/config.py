import torch

config_device_name = 'cuda:1'
config_device = torch.device(config_device_name)
config_dataset = 'AGNEWS'
attach_NE = True


class BertConfig():
    linear_layer_num = {
        'IMDB': 1,
        'SST2': 1,
        'AGNEWS': 2
    }
    dropout_rate = {
        'IMDB': 0.5,
        'SST2': 0.5,
        'AGNEWS': 0.5
    }
    is_fine_tuning = {
        'IMDB': True,
        'SST2': True,
        'AGNEWS': True
    }


class TextCNNConfig():
    channel_kernel_size = {
        'IMDB': ([50, 50, 50], [3, 4, 5]),
        'AGNEWS': ([50, 50, 50], [3, 4, 5]),
        'YAHOO': ([50, 50, 50], [3, 4, 5]),
    }
    is_static = {
        'IMDB': True,
        'AGNEWS': True,
        'YAHOO': True,
    }
    using_pretrained = {
        'IMDB': True,
        'AGNEWS': True,
        'YAHOO': False,
    }

    train_embedding_dim = {
        'IMDB': 50,
        'AGNEWS': 50,
        'YAHOO': 100,
    }


class LSTMConfig():
    num_hiddens = {
        'IMDB': 100,
        'AGNEWS': 100,
        'YAHOO': 100,
    }

    num_layers = {
        'IMDB': 2,
        'AGNEWS': 2,
        'YAHOO': 2,
    }

    is_using_pretrained = {
        'IMDB': True,
        'AGNEWS': True,
        'YAHOO': False,
    }

    word_dim = {
        'IMDB': 100,
        'AGNEWS': 100,
        'YAHOO': 100,
    }


class IMDBConfig():
    train_data_path = r'./dataset/IMDB/aclimdb/train.std'
    test_data_path = r'./dataset/IMDB/aclimdb/test.std'
    pretrained_word_vectors_path = r'./static/glove.6B.100d.txt'
    labels_num = 2
    vocab_limit_size = 80000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 230


class AGNEWSConfig():
    train_data_path = r'./dataset/AGNEWS/train.std'
    test_data_path = r'./dataset/AGNEWS/test.std'
    pretrained_word_vectors_path = r'./static/glove.6B.100d.txt'
    labels_num = 4
    vocab_limit_size = 80000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 50


class SST2Config():
    train_data_path = r'./dataset/SST2/train.std'
    test_data_path = r'./dataset/SST2/test.std'
    pretrained_word_vectors_path = r'./static/glove.6B.100d.txt'
    labels_num = 2
    vocab_limit_size = 80000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 20


config_data = {
    'IMDB': IMDBConfig,
    'AGNEWS': AGNEWSConfig,
    'SST2': SST2Config,
}


config_dataset_list = [
    'IMDB',
    'AGNEWS',
    'SST2',
]


config_pwws_use_NE = True
config_pww_NNE_attack = False

model_path = {'IMDB_Bert_MNE': 'output/train_baseline_model/2021-05-12_12:57:41/models/IMDB_Bert_0.91328_05-12-14-47.pt',
              'IMDB_Bert_replace_NE': 'output/train_baseline_model/2021-05-13_21:20:48/models/IMDB_Bert_0.91096_05-13-22-37.pt',
              'IMDB_Bert_attach_NE': 'output/train_baseline_model/2021-05-14_21:39:33/models/IMDB_Bert_0.91680_05-15-04-01.pt',
              'IMDB_Bert_attach_NE_inhance': 'output/train_baseline_model/2021-05-18_19:47:01/models/IMDB_Bert_0.91008_05-19-05-14.pt',
              'IMDB_Bert_attack_NE_weak': 'output/train_baseline_model/2021-05-19_18:03:20/models/IMDB_Bert_0.91120_05-19-23-35.pt',
              'IMDB_Bert': 'output/train_baseline_model/2021-05-11_21:36:13/models/IMDB_Bert_0.91564_05-11-22-58.pt',
              'SST2_Bert_attach_NE': 'output/train_baseline_model/2021-05-25_11:01:06/models/SST2_Bert_0.85592_05-25-11-08.pt',
              'SST2_Bert': 'output/train_baseline_model/2021-05-24_22:50:17/models/SST2_Bert_0.87078_05-24-22-59.pt',
              'AGNEWS_Bert': 'output/train_baseline_model/2021-05-25_12:16:00/models/AGNEWS_Bert_0.94250_05-25-14-51.pt',
              'AGNEWS_Bert_attach_NE': 'output/train_baseline_model/2021-05-25_17:30:41/models/AGNEWS_Bert_0.92803_05-26-05-35.pt'}


if __name__ == '__main__':
    pass
