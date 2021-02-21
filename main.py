# -*- coding:utf-8 -*-
import os
from config.dataset_config import DatasetConfig
from utils.simple_lexicon_loader import SimpleLexiconLoader
from utils.simple_lexicon_data_process import make_simple_lexicon_data
from config.sequence_tagging_config import SequenceTaggingConfig
from sequence_model.lstm_crf_model import LSTMCrfModel

config = DatasetConfig(version='resume')


def data_proc():
    make_simple_lexicon_data(config.ori_train_file, config.train_file)
    make_simple_lexicon_data(config.ori_test_file, config.test_file)
    make_simple_lexicon_data(config.ori_dev_file, config.dev_file)


def load_data(with_lexicon=False):
    loader = SimpleLexiconLoader()
    train, test, dev = loader.load_data(config.train_file, config.test_file, config.dev_file, config.lexicon_path)
    if not with_lexicon:
        train = train[0], train[2]
        test = test[0], test[2]
        dev = dev[0], dev[2]
    return train, test, dev


def train_model():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    train, test, dev = load_data(with_lexicon=False)
    tagging_config = SequenceTaggingConfig('lstm-crf')
    model = LSTMCrfModel(tagging_config)
    model.train(train, dev)
    model.save_session()


if __name__ == "__main__":
    train_model()
