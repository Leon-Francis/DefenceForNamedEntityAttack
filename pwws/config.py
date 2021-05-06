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


config_model_save_path = {
    'IMDB': r'./models/IMDB/{}_{:.5f}_{}_{}.pt',
    'AGNEWS': r'./models/AGNEWS/{}_{:.5f}_{}_{}.pt',
    'YAHOO': r'./models/YAHOO/{}_{:.5f}_{}_{}.pt',
}

config_model_lists = [
    'LSTM', 'TextCNN', 'BidLSTM', 'BidLSTM_enhanced',
    'TextCNN_enhanced', 'LSTM_enhanced', 'LSTM_adv',
    'BidLSTM_adv', 'TextCNN_adv'
]

config_dataset_list = [
    'IMDB',
    'AGNEWS',
    'YAHOO',
]

config_attack_list = [
    'PWWS',
    'TEXTFOOL',
    'RANDOM'
]

config_pwws_use_NE = True
config_RSE_mask_low = 2
config_RSE_mask_rate = 0.25


if __name__ == '__main__':
    pass