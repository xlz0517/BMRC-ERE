import torch.nn as nn
from transformers import BertModel

class BERTEncoder(nn.Module):
    def __init__(self, pretrain_path, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain_path)
        for param in self.bert.parameters():
            param.requires_grad = True


    def forward(self, token, att_mask):
        bert_output = self.bert(token, attention_mask=att_mask)
        x = bert_output[0]
        pool_output = bert_output[1]
        return x, pool_output