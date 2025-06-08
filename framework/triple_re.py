import torch
from torch import nn, optim
from tqdm import tqdm
from .dataloaders import RELoader, REDataset
from .optimization import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup

class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class Triple_RE(nn.Module):
    def __init__(self,
                 model,
                 config,
                 train,
                 val,
                 test,
                 rel2id,
                 pretrain_path,
                 ckpt,
                 batch_size=16,
                 num_workers=6,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5):

        super().__init__()
        self.config = config
        self.max_epoch = max_epoch

        self.path = train
        self.rel2id = rel2id
        self.pretrain_path = pretrain_path
        self.batch_size = batch_size

        # Load data
        self.train_loder = RELoader(
            path=train,
            rel2id=rel2id,
            pretrain_path=pretrain_path,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.val_set = REDataset(path=val, rel_dict_path=rel2id, pretrain_path=pretrain_path)
        self.test_set = REDataset(path=test, rel_dict_path=rel2id, pretrain_path=pretrain_path)

        # Model
        # self.model = nn.DataParallel(model)
        self.model = model

        # # Criterion
        # self.loss_func = nn.BCELoss()

        # # Params and optimizer 1
        params = self.model.parameters()
        self.lr = lr
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, params), lr, weight_decay=weight_decay)

        # # # Params and optimizer 2
        # config.num_train_optimization_steps = len(self.train_loder) // config.gradient_accumulation_steps * config.epoch
        # self.optimizer = self.get_bert_optimizer(self.model, config)

        # # Params and optimizer 3
        # t_total = len(self.train_loder) * config.epoch
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in self.model.named_parameters() if "bert." in n],
        #         "weight_decay": config.weight_decay_rate,
        #         "lr": config.fin_tuning_lr,
        #     },
        #     {
        #         "params": [p for n, p in self.model.named_parameters() if "bert." not in n],
        #         "weight_decay": config.weight_decay_rate,
        #         "lr": config.downs_lr,
        #     }
        # ]
        # self.optimizer = AdamW(optimizer_grouped_parameters, eps=1e-7)
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=config.warmup_prop * t_total, num_training_steps=t_total
        # )

        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

        self.flag = False

    def train_model(self, warmup=True):
        best_f1 = 0
        global_step = 0
        wait = 0
        for epoch in range(1, self.max_epoch + 1):

            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_sent_loss = AverageMeter()
            train_loder = RELoader(
                path=self.path,
                rel2id=self.rel2id,
                pretrain_path=self.pretrain_path,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=6
            )
            t = tqdm(train_loder)

            #             t = tqdm(self.train_loder)

            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass

                # sentence
                loss = self.model(*data)

                # Log
                avg_sent_loss.update(loss.item(), 1)
                t.set_postfix(sent_loss=avg_sent_loss.avg)

                # # Optimize 使用SGD学习率预热
                # if warmup == True:
                #     warmup_step = 300
                #     if global_step < warmup_step:
                #         warmup_rate = float(global_step) / warmup_step
                #     else:
                #         warmup_rate = 1.0
                #     for param_group in self.optimizer.param_groups:
                #         param_group['lr'] = self.lr * warmup_rate

                self.optimizer.zero_grad()

                #梯度回传
                loss.backward()

                #梯度裁剪
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.optimizer.step()

                #AdamW
                # self.scheduler.step()

                global_step += 1

            # Val 验证
            print("=== Epoch %d val ===" % epoch)
            self.eval()
            precision, recall, f1 = self.val_set.metric(self.model)
            if f1 > best_f1 or f1 < 1e-4:
                if f1 > 1e-4:
                    best_f1 = f1
                    print("Best ckpt and saved.")
                    torch.save({'state_dict': self.model.state_dict()}, self.ckpt)

            print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, best_f1))

            # torch.save({'state_dict': self.model.state_dict()}, 'checkpoint/epoch{}.pth.tar'.format(epoch))

            # if epoch > 1:
            #     model_file = 'checkpoint/epoch{}.pth.tar'.format(epoch-1)
            #     os.remove(model_file)

            #  学习率衰减
            if epoch > 15:
                self.config.lr = 1e-2
            #   self.config.downs_lr = self.config.downs_lr * 0.9
            if epoch > 25:
                self.config.lr = 1e-3
            if epoch > 50:
                self.config.lr = 1e-4

            # Test 测试
            print("=== Epoch %d test ===" % epoch)
            self.eval()
            precision_test, recall_test, f1_test = self.test_set.metric(self.model)
            print("\n")
        # torch.save({'state_dict': self.model.state_dict()}, 'checkpoint/final.pth.tar')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, best_f1))

    def load_state_dict(self, ckpt):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['state_dict'])

    def get_bert_optimizer(self, model, config):
        # Prepare optimizer
        # fine-tuning
        param_optimizer = list(model.named_parameters())
        # pretrain model param
        param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
        # downstream model param
        param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
        no_decay = ['bias', 'LayerNorm', 'layer_norm']
        optimizer_grouped_parameters = [
            # pretrain model param
            {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay_rate, 'lr': config.fin_tuning_lr
             },
            {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.fin_tuning_lr
             },
            # downstream model
            {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay_rate, 'lr': config.downs_lr
             },
            {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': config.downs_lr
             }
        ]
        return BertAdam(optimizer_grouped_parameters, warmup=config.warmup_prop, schedule="warmup_cosine",
                        t_total=config.num_train_optimization_steps, max_grad_norm=config.clip_grad)