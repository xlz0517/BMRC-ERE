import torch
import torch.nn as nn
import torch.nn.functional as F


class HGAT(nn.Module):
    def __init__(self, config):
        super(HGAT, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.class_nums, config.hidden_size)
        self.relation = nn.Linear(config.hidden_size, config.hidden_size)
        self.layers = nn.ModuleList([HGATLayer(config.hidden_size) for _ in range(config.hgat_layers)])

    def forward(self, x, mask):
        p = torch.arange(self.config.class_nums).long()
        if torch.cuda.is_available():
            p = p.cuda()
        p = self.relation(self.embedding(p))
        p = p.unsqueeze(0).expand(x.size(0), p.size(0), p.size(1))  # bcd
        x, p = self.gat_layer(x, p, mask)  # x bcd
        return x

    def gat_layer(self, x, p, mask=None):
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class HGATLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ra1 = RelationAttention(hidden_size)
        self.ra2 = RelationAttention(hidden_size)

    def forward(self, x, p, mask=None):
        # update sentence
        x_, a1 = self.ra1(x, p)
        # Residual connection
        x = x_ + x
        # # update relation
        p_, a2 = self.ra2(p, x, mask)
        # Residual connection
        p = p_ + p
        return x, p

# class HGATLayer(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.ra1 = RelationAttention(hidden_size)
#
#     def forward(self, x, p, mask=None):
#         # update sentence
#         x_ = self.ra1(x, p)
#         # Residual connection
#         x = x_ + x
#         return x, p

class RelationAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RelationAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(2 * hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, p, x, mask=None):
        q = self.query(p)
        k = self.key(x)
        score = self.fuse(q, k)
        if mask is not None:
            mask = 1 - mask[:, None, :].expand(-1, score.size(1), -1)
            score = score.masked_fill(mask == 1, -1e9)
        score = F.softmax(score, 2)
        v = self.value(x)
        out = torch.einsum('bcl,bld->bcd', score, v) + p
        # gate and output
        g = self.gate(torch.cat([out, p], 2)).sigmoid()
        out = g * out + (1 - g) * p
        return out, score

    def fuse(self, x, y):
        x = x.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        y = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        temp = torch.cat([x, y], 3)
        return self.score(temp).squeeze(3)

