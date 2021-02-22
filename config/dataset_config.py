# -*- coding:utf-8 -*-
import os


class DatasetConfig:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, data_name):
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.ori_train_file = os.path.join(self.data_dir, data_name, 'train.char.bmes')
        self.ori_test_file = os.path.join(self.data_dir, data_name, 'test.char.bmes')
        self.ori_dev_file = os.path.join(self.data_dir, data_name, 'dev.char.bmes')
        self.train_file = os.path.join(self.data_dir, data_name, 'train-lexicons.txt')
        self.test_file = os.path.join(self.data_dir, data_name, 'test-lexicons.txt')
        self.dev_file = os.path.join(self.data_dir, data_name, 'dev-lexicons.txt')
        self.lexicon_path = os.path.join(self.data_dir, 'records', data_name)
        if not os.path.exists(self.lexicon_path):
            os.makedirs(self.lexicon_path)
