import torch
import torch.utils.data as data
import json
import numpy as np
from random import choice
from transformers import BertTokenizer
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from configs import Config

BERT_MAX_LEN = 512


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def seq_padding(batch, padding=0):
    length_batch = [len(seq) for seq in batch]
    max_length = max(length_batch)
    return np.array([
        np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in batch
    ])


class REDataset(data.Dataset):
    def __init__(self, path, rel_dict_path, pretrain_path):
        super().__init__()
        self.config = Config()
        self.path = path
        self.data = json.load(open(path, encoding='utf-8'))

        id2rel, rel2id, rel2sub, rel2obj, rel2o, rel2s = json.load(open(rel_dict_path, encoding='utf-8'))

        id2rel = {int(i): j for i, j in id2rel.items()}
        self.num_rels = len(id2rel)
        self.id2rel = id2rel
        self.rel2id = rel2id

        self.rel2sub = rel2sub
        self.rel2obj = rel2obj
        self.rel2o = rel2o
        self.rel2s = rel2s

        self.maxlen = 100
        self.pretrain_path = pretrain_path

        self.berttokenizer = BertTokenizer.from_pretrained(pretrain_path,
                                                               do_lower_case=False,
                                                               do_basic_tokenize=False)  # do_lower_case=False

        for sent in self.data:
            ## to tuple
            triple_list = []
            for triple in sent['triple_list']:
                triple_list.append(tuple(triple))
            sent['triple_list'] = triple_list
        self.data_length = len(self.data)

        print("new loder")
        self.process_data = []
        for i in range(self.data_length):
            self.process_data += self.process(i)

    def __len__(self):
        return len(self.process_data)

    def __getitem__(self, index):
        ret = self.process_data[index]
        return ret

    def process(self, index):
        line = self.data[index]
        text = ' '.join(line['text'].split()[:self.maxlen])

        tokens = self._tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[:BERT_MAX_LEN]
        text_len = len(tokens)

        rel_set = set()

        rso = {}
        ros = {}

        for triple in line['triple_list']:
            s = triple[0]
            p = triple[1]
            o = triple[2]

            Qps = self.rel2sub[p] + " " + "[SEP] "
            stoken = self._tokenize(Qps)
            Qps_tokens = self._tokenize(Qps + line['text'])
            if len(Qps_tokens) > BERT_MAX_LEN:
                Qps_tokens = Qps_tokens[:BERT_MAX_LEN]

            Qpso = self.rel2o[p].format(s) + " " + "[SEP] "
            otoken = self._tokenize(Qpso)
            Qpso_tokens = self._tokenize(Qpso + line['text'])
            if len(Qpso_tokens) > BERT_MAX_LEN:
                Qpso_tokens = Qpso_tokens[:BERT_MAX_LEN]

            Qpo = self.rel2obj[p] + " " + "[SEP] "
            otoken1 = self._tokenize(Qpo)
            Qpo_tokens = self._tokenize(Qpo + line['text'])
            if len(Qpo_tokens) > BERT_MAX_LEN:
                Qpo_tokens = Qpo_tokens[:BERT_MAX_LEN]

            Qpos = self.rel2s[p].format(o) + " " + "[SEP] "
            stoken1 = self._tokenize(Qpos)
            Qpos_tokens = self._tokenize(Qpos + line['text'])
            if len(Qpos_tokens) > BERT_MAX_LEN:
                Qpos_tokens = Qpos_tokens[:BERT_MAX_LEN]

            ###################---------分界线-----------------#################################
            triple = (self._tokenize(triple[0])[1:-1], triple[1], self._tokenize(triple[2])[1:-1])
            rel_id = self.rel2id[triple[1]]
            rel_set.add(rel_id)

            sub_head_idx = find_head_idx(tokens, triple[0]) + len(stoken) - 2
            obj_head_idx = find_head_idx(tokens, triple[2]) + len(otoken) - 2

            obj_head_idx1 = find_head_idx(tokens, triple[2]) + len(otoken1) - 2
            sub_head_idx1 = find_head_idx(tokens, triple[0]) + len(stoken1) - 2

            if sub_head_idx != -1 and obj_head_idx != -1:
                sub_tail_idx = sub_head_idx + len(triple[0]) - 1
                obj_tail_idx = obj_head_idx + len(triple[2]) - 1
                if rel_id not in rso:
                    rso[rel_id] = []
                rso[rel_id].append(
                    (sub_head_idx, sub_tail_idx, Qps_tokens, obj_head_idx, obj_tail_idx, Qpso_tokens))

            if obj_head_idx1 != -1 and sub_head_idx1 != -1:
                obj_tail_idx1 = obj_head_idx1 + len(triple[2]) - 1
                sub_tail_idx1 = sub_head_idx1 + len(triple[0]) - 1
                if rel_id not in ros:
                    ros[rel_id] = []
                ros[rel_id].append(
                    (obj_head_idx1, obj_tail_idx1, Qpo_tokens, sub_head_idx1, sub_tail_idx1, Qpos_tokens))

        entity = []
        for triple in line['triple_list']:
            triple = (self._tokenize(triple[0])[1:-1], triple[1], self._tokenize(triple[2])[1:-1])
            sub_head_idx = find_head_idx(tokens, triple[0])
            obj_head_idx = find_head_idx(tokens, triple[2])
            if sub_head_idx != -1 and obj_head_idx != -1:
                sub_tail_idx = sub_head_idx + len(triple[0]) - 1
                obj_tail_idx = obj_head_idx + len(triple[2]) - 1
                entity.append((sub_head_idx, sub_tail_idx, obj_head_idx, obj_tail_idx))

        process = []
        if rso and ros:
            token_ids = self.berttokenizer.convert_tokens_to_ids(tokens)
            att_mask = torch.ones(len(token_ids)).long()
            rels = np.zeros(self.num_rels)

            entity_heads = np.zeros(text_len)
            entity_tails = np.zeros(text_len)

            for i in entity:
                s_start = i[0]
                s_end = i[1]
                o_start = i[2]
                o_end = i[3]

                entity_heads[s_start] = 1
                entity_heads[o_start] = 1
                entity_tails[s_end] = 1
                entity_tails[o_end] = 1

            for key in rso:
                rel_id = [key]
                rels[key] = 1

                k = choice(rso[key])
                length = len(k[2])
                sub_heads = np.zeros(length)
                sub_tails = np.zeros(length)
                Qtoken_ids = self.berttokenizer.convert_tokens_to_ids(k[2])
                Qatt_mask = torch.ones(len(Qtoken_ids)).long()
                for i in rso[key]:
                    ss, se = i[0], i[1]
                    sub_heads[ss] = 1
                    sub_tails[se] = 1

                m = choice(ros[key])
                length = len(m[2])
                obj_heads1 = np.zeros(length)
                obj_tails1 = np.zeros(length)
                Qtoken_ids1 = self.berttokenizer.convert_tokens_to_ids(m[2])
                Qatt_mask1 = torch.ones(len(Qtoken_ids1)).long()
                for i in ros[key]:
                    os1, oe1 = i[0], i[1]
                    obj_heads1[os1] = 1
                    obj_tails1[oe1] = 1

                i = choice(rso[key])
                ss, se, t1, os, oe, t2, = i[0], i[1], i[2], i[3], i[4], i[5]
                sub_head = [ss]
                sub_tail = [se]
                obj_heads = np.zeros(len(t2))
                obj_tails = np.zeros(len(t2))
                obj_heads[os] = 1
                obj_tails[oe] = 1
                Qtoken_ids_ = self.berttokenizer.convert_tokens_to_ids(t2)
                Qatt_mask_ = torch.ones(len(Qtoken_ids_)).long()

                j = choice(ros[key])
                os1, oe1, t3, ss1, se1, t4 = j[0], j[1], j[2], j[3], j[4], j[5]
                obj_head = [os1]
                obj_tail = [oe1]
                sub_heads1 = np.zeros(len(t4))
                sub_tails1 = np.zeros(len(t4))
                sub_heads1[ss1] = 1
                sub_tails1[se1] = 1
                Qtoken_ids1_ = self.berttokenizer.convert_tokens_to_ids(t4)
                Qatt_mask1_ = torch.ones(len(Qtoken_ids1_)).long()

                process.append(
                    [token_ids, att_mask, rels,
                     Qtoken_ids, Qatt_mask, Qtoken_ids_, Qatt_mask_, sub_heads, sub_tails, obj_heads, obj_tails,
                     Qtoken_ids1, Qatt_mask1, Qtoken_ids1_, Qatt_mask1_, obj_heads1, obj_tails1, sub_heads1, sub_tails1,
                     entity_heads, entity_tails,
                     rel_id, sub_head, sub_tail, obj_head, obj_tail])

        return process

    def _tokenize(self, tokens):
        re_tokens = ['[CLS]']
        for token in tokens.strip().split():
            re_tokens += self.berttokenizer.tokenize(token)
        re_tokens.append('[SEP]')
        return re_tokens

    def metric(self, model, h_bar=0.5, t_bar=0.5, exact_match=True, output_path=None):
        save_data = []
        orders = ['subject', 'relation', 'object']
        correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
        for line in tqdm(self.data):
            Pred_triples = set(self.extract_items(model, line['text'], h_bar=h_bar, t_bar=t_bar))

            Gold_triples = set(line['triple_list'])

            Pred_triples_eval, Gold_triples_eval = self.partial_match(Pred_triples,
                                                                      Gold_triples) if not exact_match else (
                Pred_triples, Gold_triples)

            correct_num += len(Pred_triples_eval & Gold_triples_eval)
            predict_num += len(Pred_triples_eval)
            gold_num += len(Gold_triples_eval)

            if output_path:
                temp = {
                    'text': line['text'],
                    'triple_list_gold': [
                        dict(zip(orders, triple)) for triple in Gold_triples
                    ],
                    'triple_list_pred': [
                        dict(zip(orders, triple)) for triple in Pred_triples
                    ],
                    'new': [
                        dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                    ],
                    'lack': [
                        dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                    ]
                }
                save_data.append(temp)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f)

        precision = correct_num / predict_num
        recall = correct_num / gold_num
        f1_score = 2 * precision * recall / (precision + recall)

        print(f'correct_num:{"%d" % correct_num}\npredict_num:{"%d" % predict_num}\ngold_num:{"%d" % gold_num}')
        print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
        return precision, recall, f1_score

    def partial_match(self, pred_set, gold_set):
        pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
                 i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
        gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
                 i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
        return pred, gold

    def extract_items(self, model, text_in, h_bar=0.5, t_bar=0.5):
        tokens = self._tokenize(text_in)

        token_ids = self.berttokenizer.convert_tokens_to_ids(tokens)
        mask = torch.ones(len(token_ids)).unsqueeze(0).long().cuda()
        if len(token_ids) > BERT_MAX_LEN:
            token_ids = token_ids[:, :BERT_MAX_LEN]
        token_ids = torch.tensor(token_ids).unsqueeze(0).long().cuda()

        if self.config.auxiliary_task:
            start_logits, end_logits = model.pre_per_entity(token_ids, mask)
            starts = np.where(start_logits[0].cpu() > h_bar)[0]
            ends = np.where(end_logits[0].cpu() > h_bar)[0]

            entity = []
            for start in starts:
                end = ends[ends >= start]
                if len(end) > 0:
                    end = end[0]
                    e = tokens[start: end + 1]
                    entity.append((e, start, end))

            entities = set()
            for e in entity:
                enti = e[0]
                enti = self.cat_wordpice(enti)
                entities.add(enti)

        relation_logits = model.pre_per_relation(token_ids, mask)
        rel_ids = np.where(relation_logits.sigmoid().cpu() > 0.5)[1]

        triple_list = []

        for r in rel_ids:
            rel_id = torch.LongTensor(np.array([[r]])).cuda()

            rel = self.id2rel[r]

            # S -> O
            Qps = self.rel2sub[rel] + " " + "[SEP] "
            Qtokens = self._tokenize(Qps + text_in)

            if len(Qtokens) > BERT_MAX_LEN:
                Qtokens = Qtokens[:BERT_MAX_LEN]
            Qtoken_ids = self.berttokenizer.convert_tokens_to_ids(Qtokens)
            Qmask = torch.ones(len(Qtoken_ids)).unsqueeze(0).long().cuda()

            if len(Qtoken_ids) > BERT_MAX_LEN:
                Qtoken_ids = Qtoken_ids[:, :BERT_MAX_LEN]
            Qtoken_ids = torch.tensor(Qtoken_ids).unsqueeze(0).long().cuda()
            sub_heads_logits, sub_tails_logits = model.pre_per_head(Qtoken_ids, Qmask)
            sub_heads = np.where(sub_heads_logits[0].cpu() > h_bar)[0]
            sub_tails = np.where(sub_tails_logits[0].cpu() > h_bar)[0]
            subjects = []
            for sub_head in sub_heads:
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    sub_tail = sub_tail[0]
                    subject = Qtokens[sub_head: sub_tail + 1]
                    subjects.append((subject, sub_head, sub_tail))

            for sub in subjects:
                s = sub[0]
                s = self.cat_wordpice(s)
                if self.config.auxiliary_task:
                    if s in entities:
                        Qpso = self.rel2o[rel].format(s) + " " + "[SEP] "
                        Qtokens_ = self._tokenize(Qpso + text_in)
                        if len(Qtokens_) > BERT_MAX_LEN:
                            Qtokens_ = Qtokens_[:BERT_MAX_LEN]
                        Qtoken_ids_ = self.berttokenizer.convert_tokens_to_ids(Qtokens_)
                        Qmask_ = torch.ones(len(Qtoken_ids_)).unsqueeze(0).long().cuda()

                        if len(Qtoken_ids_) > BERT_MAX_LEN:
                            Qtoken_ids_ = Qtoken_ids_[:, :BERT_MAX_LEN]
                        Qtoken_ids_ = torch.tensor(Qtoken_ids_).unsqueeze(0).long().cuda()

                        obj_heads_logits, obj_tails_logits = model.pre_per_tail(Qtoken_ids_, Qmask_)
                        obj_heads = np.where(obj_heads_logits[0].cpu() > t_bar)[0]
                        obj_tails = np.where(obj_tails_logits[0].cpu() > t_bar)[0]
                        objects = []

                        for obj_head in obj_heads:
                            obj_tail = obj_tails[obj_tails >= obj_head]
                            if len(obj_tail) > 0:
                                obj_tail = obj_tail[0]
                                object = Qtokens_[obj_head: obj_tail + 1]
                                objects.append((object, obj_head, obj_tail))

                        for obj in objects:
                            o = obj[0]
                            o = self.cat_wordpice(o)
                            if o in entities:
                                triple_list.append((s, rel, o))
                else:
                    Qpso = self.rel2o[rel].format(s) + " " + "[SEP] "
                    Qtokens_ = self._tokenize(Qpso + text_in)
                    if len(Qtokens_) > BERT_MAX_LEN:
                        Qtokens_ = Qtokens_[:BERT_MAX_LEN]
                    Qtoken_ids_ = self.berttokenizer.convert_tokens_to_ids(Qtokens_)
                    Qmask_ = torch.ones(len(Qtoken_ids_)).unsqueeze(0).long().cuda()

                    if len(Qtoken_ids_) > BERT_MAX_LEN:
                        Qtoken_ids_ = Qtoken_ids_[:, :BERT_MAX_LEN]
                    Qtoken_ids_ = torch.tensor(Qtoken_ids_).unsqueeze(0).long().cuda()

                    obj_heads_logits, obj_tails_logits = model.pre_per_tail(Qtoken_ids_, Qmask_)
                    obj_heads = np.where(obj_heads_logits[0].cpu() > t_bar)[0]
                    obj_tails = np.where(obj_tails_logits[0].cpu() > t_bar)[0]
                    objects = []
                    for obj_head in obj_heads:
                        obj_tail = obj_tails[obj_tails >= obj_head]
                        if len(obj_tail) > 0:
                            obj_tail = obj_tail[0]
                            object = Qtokens_[obj_head: obj_tail + 1]
                            objects.append((object, obj_head, obj_tail))
                    for obj in objects:
                        o = obj[0]
                        o = self.cat_wordpice(o)
                        triple_list.append((s, rel, o))

            if self.config.two_way:
                ################################ O -> S #################################
                Qpo = self.rel2obj[rel] + " " + "[SEP] "
                Qtokens1 = self._tokenize(Qpo + text_in)

                if len(Qtokens1) > BERT_MAX_LEN:
                    Qtokens1 = Qtokens1[:BERT_MAX_LEN]
                Qtoken_ids1 = self.berttokenizer.convert_tokens_to_ids(Qtokens1)
                Qmask1 = torch.ones(len(Qtoken_ids1)).unsqueeze(0).long().cuda()

                if len(Qtoken_ids1) > BERT_MAX_LEN:
                    Qtoken_ids1 = Qtoken_ids1[:, :BERT_MAX_LEN]
                Qtoken_ids1 = torch.tensor(Qtoken_ids1).unsqueeze(0).long().cuda()

                obj_heads_logits1, obj_tails_logits1 = model.pre_per_tail_(Qtoken_ids1, Qmask1)
                obj_heads1 = np.where(obj_heads_logits1[0].cpu() > t_bar)[0]
                obj_tails1 = np.where(obj_tails_logits1[0].cpu() > t_bar)[0]
                objects1 = []
                for obj_head in obj_heads1:
                    obj_tail = obj_tails1[obj_tails1 >= obj_head]
                    if len(obj_tail) > 0:
                        obj_tail = obj_tail[0]
                        object = Qtokens1[obj_head: obj_tail + 1]
                        objects1.append((object, obj_head, obj_tail))
                for obj in objects1:
                    o = obj[0]
                    o = self.cat_wordpice(o)
                    if self.config.auxiliary_task:
                        if o in entities:
                            Qpos = self.rel2s[rel].format(o) + " " + "[SEP] "
                            Qtokens1_ = self._tokenize(Qpos + text_in)
                            if len(Qtokens1_) > BERT_MAX_LEN:
                                Qtokens1_ = Qtokens1_[:BERT_MAX_LEN]
                            Qtoken_ids1_ = self.berttokenizer.convert_tokens_to_ids(Qtokens1_)
                            Qmask1_ = torch.ones(len(Qtoken_ids1_)).unsqueeze(0).long().cuda()

                            if len(Qtoken_ids1_) > BERT_MAX_LEN:
                                Qtoken_ids1_ = Qtoken_ids1_[:, :BERT_MAX_LEN]
                            Qtoken_ids1_ = torch.tensor(Qtoken_ids1_).unsqueeze(0).long().cuda()

                            sub_heads_logits1, sub_tails_logits1 = model.pre_per_head_(Qtoken_ids1_, Qmask1_)
                            sub_heads1 = np.where(sub_heads_logits1[0].cpu() > h_bar)[0]
                            sub_tails1 = np.where(sub_tails_logits1[0].cpu() > h_bar)[0]
                            subjects1 = []
                            for sub_head in sub_heads1:
                                sub_tail = sub_tails1[sub_tails1 >= sub_head]
                                if len(sub_tail) > 0:
                                    sub_tail = sub_tail[0]
                                    subject = Qtokens1_[sub_head: sub_tail + 1]
                                    subjects1.append((subject, sub_head, sub_tail))
                            for sub in subjects1:
                                s = sub[0]
                                s = self.cat_wordpice(s)
                                if s in entities:
                                    triple_list.append((s, rel, o))
                    else:
                        Qpos = self.rel2s[rel].format(o) + " " + "[SEP] "
                        Qtokens1_ = self._tokenize(Qpos + text_in)
                        if len(Qtokens1_) > BERT_MAX_LEN:
                            Qtokens1_ = Qtokens1_[:BERT_MAX_LEN]
                        Qtoken_ids1_ = self.berttokenizer.convert_tokens_to_ids(Qtokens1_)
                        Qmask1_ = torch.ones(len(Qtoken_ids1_)).unsqueeze(0).long().cuda()

                        if len(Qtoken_ids1_) > BERT_MAX_LEN:
                            Qtoken_ids1_ = Qtoken_ids1_[:, :BERT_MAX_LEN]
                        Qtoken_ids1_ = torch.tensor(Qtoken_ids1_).unsqueeze(0).long().cuda()

                        sub_heads_logits1, sub_tails_logits1 = model.pre_per_head_(Qtoken_ids1_, Qmask1_)
                        sub_heads1 = np.where(sub_heads_logits1[0].cpu() > h_bar)[0]
                        sub_tails1 = np.where(sub_tails_logits1[0].cpu() > h_bar)[0]
                        subjects1 = []
                        for sub_head in sub_heads1:
                            sub_tail = sub_tails1[sub_tails1 >= sub_head]
                            if len(sub_tail) > 0:
                                sub_tail = sub_tail[0]
                                subject = Qtokens1_[sub_head: sub_tail + 1]
                                subjects1.append((subject, sub_head, sub_tail))
                        for sub in subjects1:
                            s = sub[0]
                            s = self.cat_wordpice(s)
                            triple_list.append((s, rel, o))

        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        
        return list(triple_set)

    def cat_wordpice(self, x):
        new_x = []
        for i in range(len(x) - 1):
            sub_x = x[i]
            rear = x[i + 1]
            new_x.append(sub_x)
            if "##" not in rear:
                new_x.append("[blank]")
        if len(x) > 0:
            new_x.append(x[-1])
        new_x = ''.join([i.lstrip("##") for i in new_x])
        new_x = ' '.join(new_x.split('[blank]'))
        return new_x

    @staticmethod
    def collate_fn(data):
        data = list(zip(*data))

        token_ids, att_mask, rels, \
        Qtoken_ids, Qatt_mask, Qtoken_ids_, Qatt_mask_, sub_heads, sub_tails, obj_heads, obj_tails, \
        Qtoken_ids1, Qatt_mask1, Qtoken_ids1_, Qatt_mask1_, obj_heads1, obj_tails1, sub_heads1, sub_tails1, \
        entity_heads, entity_tails, \
        rel_id, sub_head, sub_tail, obj_head, obj_tail = data

        tokens_batch = torch.from_numpy(seq_padding(token_ids)).long()
        att_mask_batch = pad_sequence(att_mask, batch_first=True, padding_value=0)
        relation_batch = torch.FloatTensor(np.array(rels))

        Qtokens_batch = torch.from_numpy(seq_padding(Qtoken_ids)).long()
        Qatt_mask_batch = pad_sequence(Qatt_mask, batch_first=True, padding_value=0)
        Qtokens_batch_ = torch.from_numpy(seq_padding(Qtoken_ids_)).long()
        Qatt_mask_batch_ = pad_sequence(Qatt_mask_, batch_first=True, padding_value=0)

        Qtokens_batch1 = torch.from_numpy(seq_padding(Qtoken_ids1)).long()
        Qatt_mask_batch1 = pad_sequence(Qatt_mask1, batch_first=True, padding_value=0)
        Qtokens_batch1_ = torch.from_numpy(seq_padding(Qtoken_ids1_)).long()
        Qatt_mask_batch1_ = pad_sequence(Qatt_mask1_, batch_first=True, padding_value=0)

        sub_heads_batch = torch.from_numpy(seq_padding(sub_heads))
        sub_tails_batch = torch.from_numpy(seq_padding(sub_tails))

        obj_heads_batch = torch.from_numpy(seq_padding(obj_heads))
        obj_tails_batch = torch.from_numpy(seq_padding(obj_tails))

        obj_heads_batch1 = torch.from_numpy(seq_padding(obj_heads1))
        obj_tails_batch1 = torch.from_numpy(seq_padding(obj_tails1))

        sub_heads_batch1 = torch.from_numpy(seq_padding(sub_heads1))
        sub_tails_batch1 = torch.from_numpy(seq_padding(sub_tails1))

        entity_heads, entity_tails = torch.from_numpy(seq_padding(entity_heads)), torch.from_numpy(
            seq_padding(entity_tails))

        rel_id = torch.LongTensor(np.array(rel_id))
        sub_head = torch.LongTensor(np.array(sub_head))
        sub_tail = torch.LongTensor(np.array(sub_tail))
        obj_head = torch.LongTensor(np.array(obj_head))
        obj_tail = torch.LongTensor(np.array(obj_tail))

        return tokens_batch, att_mask_batch, relation_batch, \
               Qtokens_batch, Qatt_mask_batch, Qtokens_batch_, Qatt_mask_batch_, sub_heads_batch, sub_tails_batch, obj_heads_batch, obj_tails_batch, \
               Qtokens_batch1, Qatt_mask_batch1, Qtokens_batch1_, Qatt_mask_batch1_, obj_heads_batch1, obj_tails_batch1, sub_heads_batch1, sub_tails_batch1, \
               entity_heads, entity_tails, \
               rel_id, sub_head, sub_tail, obj_head, obj_tail


def RELoader(path, rel2id, pretrain_path, batch_size,
             shuffle, num_workers=8, collate_fn=REDataset.collate_fn):
    dataset = REDataset(path=path, rel_dict_path=rel2id, pretrain_path=pretrain_path)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader