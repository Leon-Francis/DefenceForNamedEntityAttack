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

    if_mask_NE = True

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
