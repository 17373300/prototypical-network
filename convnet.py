import torch.nn as nn
from transformers import BertModel


class PNModel(nn.Module):
    def __init__(self):
        super(PNModel, self).__init__()
        self.bert = BertModel.from_pretrained('./xs/')
        self.fc = nn.Linear(self.bert.config.hidden_size, 256)

    def forward(self, x):
        # (B,L,HD)
        x = self.bert(x).last_hidden_state
        # (B,HD)
        x = x[:, 0, :]
        # (B,256)
        x = self.fc(x)
        return x
