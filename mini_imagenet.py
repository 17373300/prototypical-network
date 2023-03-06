import os.path as osp
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer
import torch

ROOT_PATH = './materials/'


def my_collate(batch):
    sen = [item[0] for item in batch]
    label = [item[1] for item in batch]

    sen = torch.nn.utils.rnn.pad_sequence(sen,
                                          batch_first=True,
                                          padding_value=0)

    return sen, torch.LongTensor(label)


class PNDataset(Dataset):
    def __init__(self, setname):
        path = osp.join(ROOT_PATH, setname + '.json')
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
        self.sen = []
        self.label = []
        for i in data:
            self.sen.append(i['text'])
            self.label.append(i['np_label'][:8])
        self.tokenizer = BertTokenizer.from_pretrained("./xs/",
                                                       do_lower_case=False)

    def __len__(self):
        return len(self.sen)

    def __getitem__(self, i):
        token_sen = self.tokenizer.encode(self.sen[i][:500])
        return torch.LongTensor(token_sen), self.label[i]
