from torch.nn import init
from encoder.bert_encoder import BERTEncoder
from .layers import *


class MRC4RTE(nn.Module):
    def __init__(self, config):
        super(MRC4RTE, self).__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)

        self.BERT = BERTEncoder(pretrain_path=config.bert_base_case, config=config)

        self.relation = nn.Linear(hidden_size, config.class_nums)

        self.Linear1 = nn.Linear(hidden_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear3 = nn.Linear(hidden_size, hidden_size)
        self.Linear4 = nn.Linear(hidden_size, hidden_size)

        self.Linear5 = nn.Linear(hidden_size, hidden_size)
        self.Linear6 = nn.Linear(hidden_size, hidden_size)
        self.Linear7 = nn.Linear(hidden_size, hidden_size)
        self.Linear8 = nn.Linear(hidden_size, hidden_size)

        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)

        self.start_tail = nn.Linear(hidden_size, 1)
        self.end_tail = nn.Linear(hidden_size, 1)

        self.start_tail_ = nn.Linear(hidden_size, 1)
        self.end_tail_ = nn.Linear(hidden_size, 1)

        self.start_head_ = nn.Linear(hidden_size, 1)
        self.end_head_ = nn.Linear(hidden_size, 1)

        self.entity_start = nn.Linear(hidden_size, 1)
        self.entity_end = nn.Linear(hidden_size, 1)
        self.HGAT = HGAT(config)

        self.init_weights()

        self.relation_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def init_weights(self):
        self.relation.bias.data.fill_(0)
        init.xavier_uniform_(self.relation.weight, gain=1)

        self.entity_start.bias.data.fill_(0)
        init.xavier_uniform_(self.entity_start.weight, gain=1)  # initialize linear layer
        self.entity_end.bias.data.fill_(0)
        init.xavier_uniform_(self.entity_end.weight, gain=1)  # initialize linear layer

        self.start_head.bias.data.fill_(0)
        init.xavier_uniform_(self.start_head.weight, gain=1)  # initialize linear layer
        self.end_head.bias.data.fill_(0)
        init.xavier_uniform_(self.end_head.weight, gain=1)  # initialize linear layer

        self.start_tail.bias.data.fill_(0)
        init.xavier_uniform_(self.start_tail.weight, gain=1)  # initialize linear layer
        self.end_tail.bias.data.fill_(0)
        init.xavier_uniform_(self.end_tail.weight, gain=1)  # initialize linear layer

        self.start_head_.bias.data.fill_(0)
        init.xavier_uniform_(self.start_head_.weight, gain=1)  # initialize linear layer
        self.end_head_.bias.data.fill_(0)
        init.xavier_uniform_(self.end_head_.weight, gain=1)  # initialize linear layer

        self.start_tail_.bias.data.fill_(0)
        init.xavier_uniform_(self.start_tail_.weight, gain=1)  # initialize linear layer
        self.end_tail_.bias.data.fill_(0)
        init.xavier_uniform_(self.end_tail_.weight, gain=1)  # initialize linear layer

        self.Linear1.bias.data.fill_(0)
        init.xavier_uniform_(self.Linear1.weight, gain=1)  # initialize linear layer
        self.Linear2.bias.data.fill_(0)
        init.xavier_uniform_(self.Linear2.weight, gain=1)  # initialize linear layer

        self.Linear3.bias.data.fill_(0)
        init.xavier_uniform_(self.Linear3.weight, gain=1)  # initialize linear layer
        self.Linear4.bias.data.fill_(0)
        init.xavier_uniform_(self.Linear4.weight, gain=1)  # initialize linear layer

        self.Linear5.bias.data.fill_(0)
        init.xavier_uniform_(self.Linear5.weight, gain=1)  # initialize linear layer
        self.Linear6.bias.data.fill_(0)
        init.xavier_uniform_(self.Linear6.weight, gain=1)  # initialize linear layer

        self.Linear7.bias.data.fill_(0)
        init.xavier_uniform_(self.Linear7.weight, gain=1)  # initialize linear layer
        self.Linear8.bias.data.fill_(0)
        init.xavier_uniform_(self.Linear8.weight, gain=1)  # initialize linear layer

    def forward(self, token, mask, relations,
                Qtoken, Qmask, Qtoken_, Qmask_, sub_heads, sub_tails, obj_heads, obj_tails,
                Qtoken1, Qmask1, Qtoken1_, Qmask1_, obj_heads_, obj_tails_, sub_heads_, sub_tails_,
                entity_heads, entity_tails,
                rel_id, sub_head, sub_tail, obj_head, obj_tail
                ):
        # token
        hidden, pool = self.BERT(token, mask)
        batch_size, length, hidden_size = hidden.size()

        relation_logits = self.pre_relation(pool)

        entity_loss = 0
        if self.config.auxiliary_task:
            # entity
            start_logits, end_logits = self.pre_entity(hidden, mask)

            start_loss = F.binary_cross_entropy(start_logits, entity_heads.float(), reduction='none')
            start_loss = (start_loss * mask.float()).sum() / mask.float().sum()

            end_loss = F.binary_cross_entropy(end_logits, entity_tails.float(), reduction='none')
            end_loss = (end_loss * mask.float()).sum() / mask.float().sum()

            entity_loss = start_loss + end_loss

        # sub
        hidden_h, _ = self.BERT(Qtoken, Qmask)
        # obj
        q_hidden, _ = self.BERT(Qtoken_, Qmask_)

        # sub -> obj
        sub_heads_logits, sub_tails_logits = self.pre_head(hidden_h, Qmask)
        obj_heads_logits, obj_tails_logits = self.pre_tail(q_hidden, Qmask_)

        relation_loss = self.relation_criterion(relation_logits.view(-1, self.config.class_nums),
                                                relations.view(-1, self.config.class_nums))
        relation_loss = torch.sum(relation_loss) / (self.config.class_nums * batch_size)

        sub_head_loss = F.binary_cross_entropy(sub_heads_logits, sub_heads.float(), reduction='none')
        sub_head_loss = (sub_head_loss * Qmask.float()).sum() / Qmask.float().sum()

        sub_tail_loss = F.binary_cross_entropy(sub_tails_logits, sub_tails.float(), reduction='none')
        sub_tail_loss = (sub_tail_loss * Qmask.float()).sum() / Qmask.float().sum()

        obj_head_loss = F.binary_cross_entropy(obj_heads_logits, obj_heads.float(), reduction='none')
        obj_head_loss = (obj_head_loss * Qmask_.float()).sum() / Qmask_.float().sum()

        obj_tail_loss = F.binary_cross_entropy(obj_tails_logits, obj_tails.float(), reduction='none')
        obj_tail_loss = (obj_tail_loss * Qmask_.float()).sum() / Qmask_.float().sum()

        if self.config.two_way:
            # obj
            hidden_t, _ = self.BERT(Qtoken1, Qmask1)
            # sub
            q_hidden_, _ = self.BERT(Qtoken1_, Qmask1_)

            # obj -> sub
            obj_heads_logits_, obj_tails_logits_ = self.pre_tail_(hidden_t, Qmask1)
            sub_heads_logits_, sub_tails_logits_ = self.pre_head_(q_hidden_, Qmask1_)

            obj_head_loss_ = F.binary_cross_entropy(obj_heads_logits_, obj_heads_.float(), reduction='none')
            obj_head_loss_ = (obj_head_loss_ * Qmask1.float()).sum() / Qmask1.float().sum()
            obj_tail_loss_ = F.binary_cross_entropy(obj_tails_logits_, obj_tails_.float(), reduction='none')
            obj_tail_loss_ = (obj_tail_loss_ * Qmask1.float()).sum() / Qmask1.float().sum()

            sub_head_loss_ = F.binary_cross_entropy(sub_heads_logits_, sub_heads_.float(), reduction='none')
            sub_head_loss_ = (sub_head_loss_ * Qmask1_.float()).sum() / Qmask1_.float().sum()
            sub_tail_loss_ = F.binary_cross_entropy(sub_tails_logits_, sub_tails_.float(), reduction='none')
            sub_tail_loss_ = (sub_tail_loss_ * Qmask1_.float()).sum() / Qmask1_.float().sum()

            loss = self.config.rso_weight * (
                    relation_loss + sub_head_loss + sub_tail_loss + obj_head_loss + obj_tail_loss) \
                   + self.config.ros_weight * (
                           relation_loss + sub_head_loss_ + sub_tail_loss_ + obj_head_loss_ + obj_tail_loss_) \
                   + self.config.auxiliary_task_weight * entity_loss

        else:
            loss = relation_loss + sub_head_loss + sub_tail_loss + obj_head_loss + obj_tail_loss + entity_loss

        return loss

    def pre_per_relation(self, token, mask):
        hidden, pool = self.BERT(token, mask)
        pr = self.pre_relation(pool)
        return pr

    def pre_per_entity(self, token, mask):
        hidden, pool = self.BERT(token, mask)
        es, ee = self.pre_entity(hidden, mask)
        return es, ee

    def pre_per_head(self, Qtoken, Qmask):
        h, _ = self.BERT(Qtoken, Qmask)
        hs, he = self.pre_head(h, Qmask)
        return hs, he

    def pre_per_tail(self, Qtoken_, Qmask_):
        h, _ = self.BERT(Qtoken_, Qmask_)
        ts, te = self.pre_tail(h, Qmask_)
        return ts, te

    def pre_per_tail_(self, Qtoken1, Qmask1):
        h, _ = self.BERT(Qtoken1, Qmask1)
        ts, te = self.pre_tail_(h, Qmask1)
        return ts, te

    def pre_per_head_(self, Qtoken1_, Qmask1_):
        h, _ = self.BERT(Qtoken1_, Qmask1_)
        hs, he = self.pre_head_(h, Qmask1_)
        return hs, he

    def pre_relation(self, input):
        relation_input = self.dropout(input)
        r_l = self.relation(relation_input)
        return r_l

    def pre_head(self, x, mask):
        start_input = self.dropout(torch.tanh(self.Linear1(x)))
        hs = self.start_head(start_input).squeeze(2)
        hs = hs.sigmoid()
        end_input = self.dropout(torch.tanh(self.Linear2(x)))
        he = self.end_head(end_input).squeeze(2)
        he = he.sigmoid()
        return hs, he

    def pre_tail(self, x, mask):
        start_input = self.dropout(torch.tanh(self.Linear3(x)))
        ts = self.start_tail(start_input).squeeze(2)
        ts = ts.sigmoid()
        end_input = self.dropout(torch.tanh(self.Linear4(x)))
        te = self.end_tail(end_input).squeeze(2)
        te = te.sigmoid()
        return ts, te

    def pre_tail_(self, x, mask):
        start_input = self.dropout(torch.tanh(self.Linear5(x)))
        ts = self.start_tail_(start_input).squeeze(2)
        ts = ts.sigmoid()
        end_input = self.dropout(torch.tanh(self.Linear6(x)))
        te = self.end_tail_(end_input).squeeze(2)
        te = te.sigmoid()
        return ts, te

    def pre_head_(self, x, mask):
        start_input = self.dropout(torch.tanh(self.Linear7(x)))
        hs = self.start_head_(start_input).squeeze(2)
        hs = hs.sigmoid()
        end_input = self.dropout(torch.tanh(self.Linear8(x)))
        he = self.end_head_(end_input).squeeze(2)
        he = he.sigmoid()
        return hs, he

    def pre_entity(self, x, mask):
        if self.config.hgat:
            x = self.HGAT(x, mask)
        start_input = self.dropout(torch.tanh(x))
        es = self.entity_start(start_input).squeeze(2)
        es = es.sigmoid()
        end_input = self.dropout(torch.tanh(x))
        ee = self.entity_end(end_input).squeeze(2)
        ee = ee.sigmoid()
        return es, ee