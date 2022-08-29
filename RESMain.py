import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from RESCNNAttention import CNNAttention
from tool import all_metrics, tokenizer, sort_base_padding
import time

# parameter
batch_size = 16
times = time.ctime(time.time())

# 数据准备
data = {}
# 加载数据,data中存有所有智能合约的SBT序列{'idx':'contract'}
with open(r'dataset/data_sbt.jsonl') as f:
    for line in f:
        line = line.strip()
        js = json.loads(line)
        data[js['idx']] = js['contract'].replace('\n', '')
print('数据载入完成')

train_labels = []
with open(r'dataset/train.txt') as f:
    for line in f:
        line = line.strip()
        url, label = line.split('\t')
        train_labels.append([url, int(label)])
print('训练标签载入完成')

test_labels = []
with open(r'dataset/test.txt') as f:
    for line in f:
        line = line.strip()
        url, label = line.split('\t')
        test_labels.append([url, int(label)])
print('测试标签载入完成')


class MyDataSet(Dataset):
    def __init__(self, data, labels):
        super(MyDataSet, self).__init__()
        self.sorted_idx_list = sort_base_padding(data=data, labels=labels)
        self.data = data
        self.labels = labels

    def __len__(self):
        # 所有enc_inputs的数量
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[self.sorted_idx_list[idx]], self.labels[self.sorted_idx_list[idx]]


train_dataset = MyDataSet(data, train_labels)
test_dataset = MyDataSet(data, test_labels)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# embedding_dim 内部词向量大小 对于输出维度没有影响
CNNATTENTION = CNNAttention(d_model=200, ndead=20, batch_first=True, num_layers=4, input_dim=1000, out_dim=2)

# cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
device_ids = [0]
model = torch.nn.DataParallel(CNNATTENTION, device_ids=device_ids)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-5)

criterion = nn.CrossEntropyLoss().cuda()

loss_list = []


def train(epoch):
    global loss_list
    loss_num = 0.0
    for index, (inputs, target) in enumerate(train_loader):
        token = tokenizer(list(inputs), padding=True, return_tensors='pt')
        ids = token['input_ids']
        ids, target = ids.to(device), target.to(device)
        outputs = model(ids)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_num += loss.item()

        if index % 400 == 399:
            print(f'epoch:{epoch + 1},index:{index + 1},loss:{loss_num / 400:.4f}')
            loss_list.append(loss_num / 400)
            loss_num = 0


def test():
    global loss_list
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, label in test_loader:
            token = tokenizer(list(images), padding=True, return_tensors='pt')
            ids = token['input_ids']
            ids, label = ids.to(device), label.to(device)
            outputs = model(ids)
            _, predicted = torch.max(outputs.data, dim=1)
            all_preds.extend(predicted)
            all_labels.extend(label)

        tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
        f1, precision, recall, mcc, g_mean, tp, tn, fp, fn, auc, acc = all_metrics(tensor_labels, tensor_preds)
        loss_value = sum(loss_list) / len(loss_list)
        loss_list = []
        print(
            f'f1: {round(f1, 4)}, precision: {round(precision, 4)}, recall: {round(recall, 4)}, auc: {auc}, mcc: {mcc}, g_mean: {g_mean}, acc: {acc}, loss: {loss_value}')
        print('tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))

        reslust = [f1, precision, recall, mcc, g_mean, tp, tn, fp, fn, auc, acc, loss_value]
        save_data.append(reslust)


if __name__ == '__main__':
    save_data = []
    times = time.ctime(time.time())
    for epoch in range(50):
        time_begin_train = time.time()
        train(epoch)
        print(time.time() - time_begin_train, 1)
        print('-' * 50)
        test()

    save_df = pd.DataFrame(data=save_data,
                           columns=['f1', 'precision', 'recall', 'mcc', 'g_mean', 'tp', 'tn', 'fp', 'fn', 'auc', 'acc',
                                    'loss'])
    save_df.to_excel(fr'./Result/{times}.xlsx', index=False)
    print(time.ctime(time.time()) - times)
