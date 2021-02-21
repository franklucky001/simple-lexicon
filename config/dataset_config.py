# -*- coding:utf-8 -*-
import os


class DatasetConfig:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, version='v1'):
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.ori_train_file = os.path.join(self.data_dir, version, 'train.char.bmes')
        self.ori_test_file = os.path.join(self.data_dir, version, 'test.char.bmes')
        self.ori_dev_file = os.path.join(self.data_dir, version, 'dev.char.bmes')
        self.train_file = os.path.join(self.data_dir, version, 'train-lexicons.txt')
        self.test_file = os.path.join(self.data_dir, version, 'test-lexicons.txt')
        self.dev_file = os.path.join(self.data_dir, version, 'dev-lexicons.txt')
        self.lexicon_path = os.path.join(self.data_dir, 'records', version)
        if not os.path.exists(self.lexicon_path):
            os.makedirs(self.lexicon_path)
