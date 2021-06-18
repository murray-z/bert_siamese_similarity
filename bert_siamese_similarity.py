# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from config import config


class BertSiamese(nn.Module):
    def __init__(self):
        super(BertSiamese, self).__init__()
        bert_name = config["bert_name"]
        bert_config = BertConfig.from_pretrained(bert_name)
        self.bert = BertModel.from_pretrained(bert_name, config=bert_config)
        hidden_size = bert_config.hidden_size
        dropout = bert_config.hidden_dropout_prob
        self.fc = nn.Linear(hidden_size, config["hidden"])
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        cls = F.relu(self.drop(outputs[1]))
        out = self.fc(cls)
        return out


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """0表示相似"""
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.), 2)
        )
        return loss_contrastive
