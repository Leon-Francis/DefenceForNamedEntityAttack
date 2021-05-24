from baseline_data import IMDB_Dataset, SST2_Dataset, AGNEWS_Dataset
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from baseline_model import Baseline_Bert
from baseline_tools import logging, get_time
from baseline_config import Baseline_Config, config_path, dataset_config
from datetime import datetime
import os
from shutil import copyfile
import copy


def save_config(path):
    copyfile(config_path, path + r'/config.txt')


def build_dataset():
    if Baseline_Config.dataset == 'IMDB':
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          if_mask_NE=Baseline_Config.if_mask_NE,
                                          if_replace_NE=Baseline_Config.if_replace_NE,
                                          if_attach_NE=Baseline_Config.if_attach_NE,
                                          debug_mode=Baseline_Config.debug_mode)
        test_dataset_orig = IMDB_Dataset(train_data=False,
                                         if_mask_NE=Baseline_Config.if_mask_NE,
                                         if_replace_NE=Baseline_Config.if_replace_NE,
                                         if_attach_NE=Baseline_Config.if_attach_NE,
                                         debug_mode=Baseline_Config.debug_mode)
    elif Baseline_Config.dataset == 'SST2':
        train_dataset_orig = SST2_Dataset(train_data=True,
                                          if_attach_NE=Baseline_Config.if_attach_NE,
                                          debug_mode=Baseline_Config.debug_mode)
        test_dataset_orig = SST2_Dataset(train_data=False,
                                         if_attach_NE=Baseline_Config.if_attach_NE,
                                         debug_mode=Baseline_Config.debug_mode)
    elif Baseline_Config.dataset == 'AGNEWS':
        train_dataset_orig = AGNEWS_Dataset(train_data=True,
                                            if_attach_NE=Baseline_Config.if_attach_NE,
                                            debug_mode=Baseline_Config.debug_mode)
        test_dataset_orig = AGNEWS_Dataset(train_data=False,
                                           if_attach_NE=Baseline_Config.if_attach_NE,
                                           debug_mode=Baseline_Config.debug_mode)
    train_data = DataLoader(train_dataset_orig,
                            batch_size=Baseline_Config.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Baseline_Config.batch_size,
                           shuffle=False,
                           num_workers=4)
    return train_data, test_data


def train(train_data, baseline_model, criterion, optimizer):
    loss_mean = 0.0
    for x, y in train_data:
        x, y = x.to(Baseline_Config.train_device), y.to(
            Baseline_Config.train_device)
        logits = baseline_model(x)
        loss = criterion(logits, y)
        loss_mean += loss.item()
        if loss.item() > Baseline_Config.skip_loss:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_mean / len(train_data)


@torch.no_grad()
def evaluate(test_data, baseline_model, criterion):
    loss_mean = 0.0
    correct = 0
    total = 0
    for x, y in test_data:
        x, y = x.to(Baseline_Config.train_device), y.to(
            Baseline_Config.train_device)
        logits = baseline_model(x)
        loss = criterion(logits, y)
        loss_mean += loss.item()

        predicts = logits.argmax(dim=-1)
        correct += predicts.eq(y).float().sum().item()
        total += y.size()[0]

    return loss_mean / len(test_data), correct / total


if __name__ == '__main__':
    logging('Using cuda device gpu: ' + str(Baseline_Config.cuda_idx))
    cur_dir = Baseline_Config.output_dir + '/train_baseline_model/' + datetime.now(
    ).strftime("%Y-%m-%d_%H:%M:%S")
    cur_models_dir = cur_dir + '/models'
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_models_dir)

    logging('Saving into directory ' + cur_dir)
    save_config(cur_dir)

    logging('preparing data...')
    train_data, test_data = build_dataset()

    logging('init models, optimizer, criterion...')
    baseline_model = Baseline_Bert(
        label_num=dataset_config[Baseline_Config.dataset].labels_num,
        linear_layer_num=Baseline_Config.linear_layer_num,
        dropout_rate=Baseline_Config.dropout_rate,
        is_fine_tuning=Baseline_Config.is_fine_tuning).to(
            Baseline_Config.train_device)

    optimizer = optim.AdamW([{
        'params': baseline_model.bert_model.parameters(),
        'lr': Baseline_Config.Bert_lr
    }, {
        'params': baseline_model.fc.parameters()
    }],
        lr=Baseline_Config.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.85,
                                                     patience=3,
                                                     verbose=True,
                                                     min_lr=3e-8)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                   lr_lambda=lambda ep: 1e-2
                                                   if ep < 4 else 1.0)

    criterion = nn.CrossEntropyLoss().to(Baseline_Config.train_device)

    logging('Start training...')
    best_acc = 0.0
    temp_path = cur_models_dir + \
        f'/{Baseline_Config.dataset}_{Baseline_Config.baseline}_temp_model.pt'
    for ep in range(Baseline_Config.epoch):
        logging(f'epoch {ep} start train')
        train_loss = train(train_data, baseline_model, criterion, optimizer)
        logging(f'epoch {ep} start evaluate')
        evaluate_loss, acc = evaluate(test_data, baseline_model, criterion)
        if acc > best_acc:
            best_acc = acc
            best_path = cur_models_dir + \
                f'/{Baseline_Config.dataset}_{Baseline_Config.baseline}_{acc:.5f}_{get_time()}.pt'
            best_state = copy.deepcopy(baseline_model.state_dict())

            if ep > 3 and best_acc > Baseline_Config.save_acc_limit and best_state != None:
                logging(f'saving best model acc {best_acc:.5f} in {temp_path}')
                torch.save(best_state, temp_path)

        warmup_scheduler.step(ep + 1)
        scheduler.step(evaluate_loss, ep + 1)

        logging(
            f'epoch {ep} done! train_loss {train_loss:.5f} evaluate_loss {evaluate_loss:.5f} \n'
            f'acc {acc:.5f} now best_acc {best_acc:.5f}')

    if best_acc > Baseline_Config.save_acc_limit and best_state != None:
        logging(f'saving best model acc {best_acc:.5f} in {best_path}')
        torch.save(best_state, best_path)
