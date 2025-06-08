1、训练和测试
#执行下述脚本训练并测试模型,训练的哪个数据集在train.py中可调整，超参数在configs.py中可调整
```
python train.py
```

2、文件说明
/datasets/bert：存放预训练语言模型及其相关配置文件
/datasets/data：存放数据集
/encoder/bert_encoder.py：编码器脚本代码文件
/framework/dataloaders.py：数据预处理脚本
/framework/optimation.py：优化器封装将本
/framework/triple_re.py：框架封装脚本
/models/layers.py：通用神经网络层封装脚本
/models/MRC4RE：模型脚本
configs.py：参数配置文件脚本
train.py：训练模型入口脚本
utils.py：工具类脚本

3、如何下载bert
git clone https://huggingface.co/bert-base-cased
git clone https://huggingface.co/bert-base-uncased

cd bert-base-cased
git lfs pull
cd ..
cd bert-base-uncased
git lfs pull
