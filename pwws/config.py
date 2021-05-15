import torch

config_device_name = 'cuda:0'
config_device = torch.device(config_device_name)
config_dataset = 'IMDB'


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
    train_data_path = r'./dataset/IMDB/train_standard.txt'
    test_data_path = r'./dataset/IMDB/test_standard.txt'
    pretrained_word_vectors_path = r'./static/glove.6B.100d.txt'
    labels_num = 2
    vocab_limit_size = 80000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 230
    clean_1k_path = r'./static/IMDB/clean1k.txt'
    adv10_path = r'./dataset/IMDB/train_adv10.txt'
    adv_train_path = {
        'LSTM': r'./static/IMDB/LSTM_adv.txt',
        'BidLSTM': r'./static/IMDB/BidLSTM_adv.txt',
        'TextCNN': r'./static/IMDB/TextCNN_adv.txt',
    }
    syn_path = r'./static/IMDB/synonymous.csv'


class AGNEWSConfig():
    train_data_path = r'./dataset/AGNEWS/train_standard.txt'
    test_data_path = r'./dataset/AGNEWS/test_standard.txt'
    pretrained_word_vectors_path = r'./static/glove.6B.100d.txt'
    labels_num = 4
    vocab_limit_size = 80000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 50
    clean_1k_path = r'./static/AGNEWS/clean1k.txt'
    adv10_path = r'./dataset/AGNEWS/train_adv10.txt'
    adv_train_path = {
        'LSTM': r'./static/AGNEWS/LSTM_adv.txt',
        'BidLSTM': r'./static/AGNEWS/BidLSTM_adv.txt',
        'TextCNN': r'./static/AGNEWS/TextCNN_adv.txt',
    }
    syn_path = r'./static/AGNEWS/synonymous.csv'


class YAHOOConfig():
    train_data_path = r'./dataset/YAHOO/train150k_standard.txt'
    test_data_path = r'./dataset/YAHOO/test5k_standard.txt'
    pretrained_word_vectors_path = r'./static/glove.6B.100d.txt'
    labels_num = 10
    vocab_limit_size = 80000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 100
    clean_1k_path = r'./static/YAHOO/clean1k.txt'
    adv10_path = r'./dataset/YAHOO/train_adv10.txt'
    adv_train_path = {
        'LSTM': r'./static/YAHOO/LSTM_adv.txt',
        'BidLSTM': r'./static/YAHOO/BidLSTM_adv.txt',
        'TextCNN': r'./static/YAHOO/TextCNN_adv.txt',
    }
    syn_path = r'./static/YAHOO/synonymous.csv'


config_data = {
    'IMDB': IMDBConfig,
    'AGNEWS': AGNEWSConfig,
    'YAHOO': YAHOOConfig,
}


config_dataset_list = [
    'IMDB',
    'AGNEWS',
    'YAHOO',
]


config_pwws_use_NE = True
config_pww_NNE_attack = False

model_path = {'IMDB_Bert_MNE': 'output/train_baseline_model/2021-05-12_12:57:41/models/IMDB_Bert_0.91328_05-12-14-47.pt',
              'IMDB_Bert_replace_NE': 'output/train_baseline_model/2021-05-13_21:20:48/models/IMDB_Bert_0.91096_05-13-22-37.pt',
              'IMDB_Bert_attach_NE': 'output/train_baseline_model/2021-05-14_21:39:33/models/IMDB_Bert_0.91680_05-15-04-01.pt',
              'IMDB_Bert': 'output/train_baseline_model/2021-05-11_21:36:13/models/IMDB_Bert_0.91564_05-11-22-58.pt'}


if __name__ == '__main__':
    pass
