import os
import sys
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoTokenizer
import json


def all_metrics(y_true, y_pred, is_training=False) -> torch.Tensor:
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    acc = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    mcc_up = (tp * tn - tp * fn)
    mcc_below = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_up / mcc_below
    g_mean = (precision * recall) ** 0.5

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training

    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = -1

    return f1.item(), precision.item(), recall.item(), mcc.item(), g_mean.item(), tp.item(), tn.item(), fp.item(), fn.item(), auc , acc.item()


tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

def sort_base_padding(data,labels):

    def key_sort_function(x):
        x = data[x]
        token_length = len(tokenizer(x)['input_ids'])
        return token_length

    idx_list = list(labels.keys())

    # aim_list:[(data,label:str),]
    idx_list.sort(key=key_sort_function)

    return idx_list

if __name__ == '__main__':

    data = {}
    # 加载数据,data中存有所有智能合约的源代码{'idx':'contract'}
    with open(r'dataset/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data[js['idx']] = js['contract'].replace('\n', '')
    print('数据载入完成')

    sorted_idx_list = sort_base_padding(data=data)

