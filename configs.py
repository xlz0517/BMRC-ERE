import os


class Config(object):
    def __init__(self):

        # Ablation experiment start
        # 实体抽取部分是否双向
        self.two_way = True
        # 是否开启实体抽取辅助任务
        self.auxiliary_task = True
        # 是否使用异质图注意力网络
        self.hgat = True
        # 异质图注意力网络的层数设置
        self.hgat_layers = 2
        # 是否使用数据增强
        self.augmentation = True
        # Ablation experiment end

        self.batch_size = 2
        self.epoch = 50

        self.max_length = 100

        #[0.5,0.3,0.2], [0.4,0.4,0.2], [0.7,0.3,1]
        self.rso_weight = 0.5
        self.ros_weight = 0.3
        self.auxiliary_task_weight = 0.2

        # SGD settings
        self.lr = 1e-1
        self.weight_decay = 1e-5 #CONLL04:1e-5, ACE05:1e-6

        # bert adam start
        self.fin_tuning_lr = 1e-5
        self.downs_lr = 2 * (1e-5)
        self.weight_decay_rate = 0.01
        self.warmup_prop = 0.1
        self.clip_grad = 2.
        self.num_train_optimization_steps = 100
        self.gradient_accumulation_steps = 1
        # bert adam end

        # linear dropout
        # [0.1, 0.2, 0.3]
        self.dropout = 0.1

        # early stopping patience level
        self.patience = 9
        self.training_criteria = 'micro_f1' # or 'macro_f1'

        # hidden_size
        self.hidden_size = 768

        # relation class num
        self.nyt_class = 24
        self.webnlg_class = 171
        self.conll04_class = 5
        self.ace05_class = 6

        self.class_nums = None
        self.seed = 2023

        self.pool_type = 'avg'
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')

        self.webnlg_ckpt = 'checkpoint/webnlg.pth.tar'
        self.conll04_ckpt = 'checkpoint/conll04.pth.tar'
        self.ace05_ckpt = 'checkpoint/ace05.pth.tar'

        root_path = 'datasets'
        self._get_path(root_path)

    def _get_path(self, root_path):
        self.root_path = root_path

        # bert base uncase bert\bert-base-uncased
        self.bert_base = os.path.join(root_path, 'bert/bert-base-uncased')
        self.bert_base_case = os.path.join(root_path, 'bert/bert-base-cased')

        # conll04-triple
        self.conll04_rel2id = os.path.join(root_path, 'data/CONLL04/rel2id.json')
        if self.augmentation:
            self.conll04_train = os.path.join(root_path, 'data/CONLL04/train_triples_augmentation.json')
        else:
            self.conll04_train = os.path.join(root_path, 'data/CONLL04/train_triples.json')
        self.conll04_val = os.path.join(root_path, 'data/CONLL04/dev_triples.json')
        self.conll04_test = os.path.join(root_path, 'data/CONLL04/test_triples.json')

        # ace05-triple
        self.ace05_rel2id = os.path.join(root_path, 'data/ACE05/rel2id.json')
        if self.augmentation:
            self.ace05_train = os.path.join(root_path, 'data/ACE05/train_triples_augmentation.json')
        else:
            self.ace05_train = os.path.join(root_path, 'data/ACE05/train_triples.json')
        self.ace05_val = os.path.join(root_path, 'data/ACE05/dev_triples.json')
        self.ace05_test = os.path.join(root_path, 'data/ACE05/test_triples.json')

        # webnlg-triple
        self.webnlg_rel2id = os.path.join(root_path, 'data/webnlg/rel2id.json')
        if self.augmentation:
            self.webnlg_train = os.path.join(root_path, 'data/webnlg/train_triples_augmentation.json')
        else:
            self.webnlg_train = os.path.join(root_path, 'data/webnlg/train_triples.json')
        self.webnlg_val = os.path.join(root_path, 'data/webnlg/dev_triples.json')
        self.webnlg_test = os.path.join(root_path, 'data/webnlg/test_triples.json')