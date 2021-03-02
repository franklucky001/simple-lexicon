# -*- coding:utf-8 -*-
import sys
from config.dataset_config import DatasetConfig
from utils.simple_lexicon_loader import SimpleLexiconLoader
from utils.simple_lexicon_data_process import make_simple_lexicon_data


def convert_data():
    make_simple_lexicon_data(config.ori_train_file, config.train_file)
    make_simple_lexicon_data(config.ori_test_file, config.test_file)
    make_simple_lexicon_data(config.ori_dev_file, config.dev_file)


def init_data_loader():
    loader = SimpleLexiconLoader(use_cache=False)
    loader.load_data(config.train_file, config.test_file, config.dev_file, config.lexicon_path)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        data_name = 'resume'
    else:
        data_name = sys.argv[1]
    config = DatasetConfig(data_name=data_name)
    convert_data()
    init_data_loader()
