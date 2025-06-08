from models.MRC4RE import MRC4RTE
from framework.triple_re import Triple_RE
from configs import Config
from utils import count_params
import numpy as np
import torch
import random, argparse

torch.cuda.set_device(0)


def seed_torch(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--dataset', default='conll04', type=str,
                        help='specify the dataset from ["webnlg","conll04","ace05"]')
    args = parser.parse_args()
    dataset = args.dataset
    is_train = args.train
    config = Config()
    if config.seed is not None:
        print(config.seed)
        seed_torch(config.seed)
    if dataset == 'webnlg':
        print('train--' + dataset + config.webnlg_ckpt)
        config.class_nums = config.webnlg_class
        model = MRC4RTE(config)
        count_params(model)
        framework = Triple_RE(model,
                              config,
                              train=config.webnlg_train,
                              val=config.webnlg_val,
                              test=config.webnlg_test,
                              rel2id=config.webnlg_rel2id,
                              pretrain_path=config.bert_base_case,
                              ckpt=config.webnlg_ckpt,
                              batch_size=config.batch_size,
                              max_epoch=config.epoch,
                              lr=config.lr,
                              num_workers=4)
        framework.train_model()
        # framework.load_state_dict(config.webnlg_ckpt)
        # print('test:' + config.webnlg_ckpt)
        # framework.test_set.metric(framework.model)
    elif dataset == 'conll04':
        print('train--' + dataset)
        config.class_nums = config.conll04_class
        model = MRC4RTE(config)
        count_params(model)
        # # 如果训练中断，加载保存的模型参数继续训练
        # checkpoint = torch.load(config.conll04_ckpt)
        # model.load_state_dict(checkpoint['state_dict'])
        framework = Triple_RE(model,
                              config,
                              train=config.conll04_train,
                              val=config.conll04_val,
                              test=config.conll04_test,
                              rel2id=config.conll04_rel2id,
                              pretrain_path=config.bert_base_case,
                              ckpt=config.conll04_ckpt,
                              batch_size=config.batch_size,
                              max_epoch=config.epoch,
                              lr=config.lr)
        framework.train_model()
        # framework.load_state_dict(config.conll04_ckpt)
        # print('test:' + config.conll04_ckpt)
        # framework.test_set.metric(framework.model)
        # output_path = 'save_result/conll04_result.json'
        # framework.test_set.metric(framework.model, output_path=output_path)
    elif dataset == 'ace05':
        print('train--' + dataset)
        config.class_nums = config.ace05_class
        model = MRC4RTE(config)
        count_params(model)
        # # 如果训练中断，加载保存的模型参数继续训练
        # checkpoint = torch.load(config.ace05_ckpt)
        # model.load_state_dict(checkpoint['state_dict'])
        framework = Triple_RE(model,
                              config,
                              train=config.ace05_train,
                              val=config.ace05_val,
                              test=config.ace05_test,
                              rel2id=config.ace05_rel2id,
                              pretrain_path=config.bert_base_case,
                              ckpt=config.ace05_ckpt,
                              batch_size=config.batch_size,
                              max_epoch=config.epoch,
                              lr=config.lr,
                              weight_decay=config.weight_decay)
        framework.train_model()
        # framework.load_state_dict(config.ace05_ckpt)
        # print('test:' + config.ace05_ckpt)
        # framework.test_set.metric(framework.model)
    else:
        print('unkonw dataset')