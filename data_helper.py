# -*- coding: UTF-8 -*-

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config import config

class SiameseDataSet(Dataset):
    def __init__(self, data_path):
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_name"])
        text1 = []
        text2 = []
        label = []
        with open(data_path) as f:
            for line in f:
                lis = line.strip().split("\t")
                text1.append(lis[0])
                text2.append(lis[1])
                label.append(int(lis[2]))

        self.input_ids_1, self.token_type_ids_1, self.attention_mask_1 = self._encoder(text1)
        self.input_ids_2, self.token_type_ids_2, self.attention_mask_2 = self._encoder(text2)
        self.label = torch.tensor(label, dtype=torch.int)

    def _encoder(self, texts):
        input_ids, token_type_ids, attention_mask = [], [], []
        for text in texts:
            res = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=config["max_seq_len"], pad_to_max_length=True)
            input_ids.append(res['input_ids'])
            token_type_ids.append(res['token_type_ids'])
            attention_mask.append(res['attention_mask'])
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.input_ids_1[item], self.attention_mask_1[item], self.token_type_ids_1[item], \
               self.input_ids_2[item], self.attention_mask_2[item], self.token_type_ids_2[item], self.label[item]