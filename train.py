# -*- coding: UTF-8 -*-


import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from bert_siamese_similarity import BertSiamese, ContrastiveLoss
from data_helper import SiameseDataSet
from config import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = config["batch_size"]
lr = config["lr"]
epochs = config["epochs"]
save_model_path = config["save_model_path"]
criterion = ContrastiveLoss()
train_data = "./data/train.txt"
dev_data = "./data/dev.txt"
test_data = "./data/test.txt"

def dev(model, data_loader):
    dev_loss = 0.
    step = 0.
    model.dev()
    with torch.no_grad():
        for i, batch in enumerate(data_loader, start=1):
            batch = [d.to(device) for d in batch]
            output1 = model(*batch[0:3])
            output2 = model(*batch[3:6])
            label = batch[-1]
            loss = criterion(output1, output2, label)
            dev_loss += loss.item()
            step += 1
    return dev_loss/step

def train():
    # 加载数据集
    train_dataloader = DataLoader(SiameseDataSet(train_data), batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(SiameseDataSet(dev_data), batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(SiameseDataSet(test_data), batch_size=batch_size, shuffle=False)

    # 加载模型，优化器
    model = BertSiamese()
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # 开始训练
    best_dev_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        for i, batch in enumerate(train_dataloader, start=1):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            output1 = model(*batch[0:3])
            output2 = model(*batch[3:6])
            label = batch[-1]
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Train epoch:{} step:{} loss:{}".format(epoch, i, loss.item()))

        # dev
        dev_loss = dev(model, dev_dataloader)
        print("Dev  epoch:{} loss:{}".format(epoch, dev_loss))
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), save_model_path)
            best_dev_loss = dev_loss

if __name__ == "__main__":
    train()
