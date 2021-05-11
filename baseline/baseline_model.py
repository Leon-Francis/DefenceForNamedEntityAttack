import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


class Baseline_Bert(nn.Module):
    def __init__(self,
                 label_num: int,
                 linear_layer_num: int,
                 dropout_rate: float,
                 is_fine_tuning=True):
        super(Baseline_Bert, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = 768
        if not is_fine_tuning:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        modules = [nn.Dropout(dropout_rate)]

        for i in range(linear_layer_num - 1):
            modules += [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
        modules.append(nn.Linear(self.hidden_size, label_num))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        # inputs = (x, types, masks)
        encoder, pooled = self.bert_model(x)[:]
        logits = self.fc(pooled)
        return logits

    def predict_prob(self, X: torch.Tensor, y_true: torch.Tensor):
        if X.dim() == 1:
            X = X.view(1, -1)
        if y_true.dim() == 0:
            y_true = y_true.view(1)

        with torch.no_grad():
            encoder, pooled = self.bert_model(X)[:]
            logits = self.fc(pooled)
            logits = F.softmax(logits, dim=1)
            prob = [logits[i][y_true[i]].item() for i in range(y_true.size(0))]
            return prob

    def predict_class(self, X: torch.Tensor):
        if X.dim() == 1:
            X = X.view(1, -1)
        predicts = None
        with torch.no_grad():
            encoder, pooled = self.bert_model(X)[:]
            logits = self.fc(pooled)
            logits = F.softmax(logits, dim=1)
            predicts = [one.argmax(0).item() for one in logits]
        return predicts
