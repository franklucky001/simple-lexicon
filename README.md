# simple-lexicon
Chinese ner, lstm-crf with tf implement simple lexicon

## 数据准备, 配置分词器
在simple_lexicon_data_process.py文件配置分词器
在preprocessor中设置use_cache=False生成训练record文件
loader = SimpleLexiconLoader(use_cache=False)
python preprocessor.py

## 训练
build docker
设置环境变量
```shell
MODEL_NAME="lstm-lexicon-crf"
DATA_NAME="resume"
USE_LEXICON="True"
```
run sh docker_exec.sh

## 预测推理 （待开发）
override SimpleLexiconModel中的predict函数加载record目录的词典生成
预处理后的数据导入session即可