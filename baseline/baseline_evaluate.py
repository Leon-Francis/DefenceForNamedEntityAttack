from baseline_data import IMDB_Dataset
import torch
from torch.utils.data import DataLoader
from torch import nn
from baseline_model import Baseline_Bert
from baseline_tools import logging
from baseline_config import Baseline_Config, dataset_config, model_path
from tqdm import tqdm

evaluate_device = torch.device('cuda:1')


def build_dataset(if_mask_NE, if_replace_NE, if_attach_NE):
    test_dataset_orig = IMDB_Dataset(train_data=False,
                                     if_mask_NE=if_mask_NE,
                                     if_replace_NE=if_replace_NE,
                                     if_attach_NE=if_attach_NE,
                                     debug_mode=False)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Baseline_Config.batch_size,
                           shuffle=False,
                           num_workers=4)
    return test_data


@torch.no_grad()
def evaluate(test_data, baseline_model, criterion):
    loss_mean = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(test_data):
        x, y = x.to(evaluate_device), y.to(
            evaluate_device)
        logits = baseline_model(x)
        loss = criterion(logits, y)
        loss_mean += loss.item()

        predicts = logits.argmax(dim=-1)
        correct += predicts.eq(y).float().sum().item()
        total += y.size()[0]

    return loss_mean / len(test_data), correct / total


if __name__ == '__main__':

    if_mask_NE = True
    if_replace_NE = False
    if_attach_NE = False

    test_data = build_dataset(if_mask_NE, if_replace_NE, if_attach_NE)

    baseline_model = Baseline_Bert(
        label_num=dataset_config[Baseline_Config.dataset].labels_num,
        linear_layer_num=Baseline_Config.linear_layer_num,
        dropout_rate=Baseline_Config.dropout_rate,
        is_fine_tuning=Baseline_Config.is_fine_tuning).to(
            evaluate_device)

    baseline_model.load_state_dict(
        torch.load(model_path['IMDB_Bert_MNE'], map_location=evaluate_device))

    criterion = nn.CrossEntropyLoss().to(evaluate_device)

    evaluate_loss, acc = evaluate(test_data, baseline_model, criterion)

    logging(f'acc = {acc}')
