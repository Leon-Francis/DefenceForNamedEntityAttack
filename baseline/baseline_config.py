config_path = './baseline/baseline_config'


class Baseline_Config:
    output_dir = '.output'
    cuda_idx = 0
    train_device = torch.device('cuda:' + str(cuda_idx))
    batch_size = 32
    dataset = 'IMDB'

    if_mask_NE = True

    linear_layer_num = 3
    dropout_rate = 0.3
    is_fine_tuning = True

    Bert_lr = 1e-5
    lr = 1e-3
    skip_loss = 0.16


dataset_config = {'IMDB': IMDBConfig}


class IMDBConfig():
    train_data_path = r'./dataset/IMDB/aclImdb/train.std'
    test_data_path = r'./dataset/IMDB/aclImdb/test.std'
    labels_num = 2
    tokenizer_type = 'bert'
    remove_stop_words = False
    sen_len = 230
    vocab_size = bert_vocab_size