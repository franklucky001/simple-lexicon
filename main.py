# -*- coding:utf-8 -*-
import os
from config.dataset_config import DatasetConfig
from utils.simple_lexicon_loader import SimpleLexiconLoader
from utils.simple_lexicon_data_process import make_simple_lexicon_data
from config.sequence_tagging_config import SequenceTaggingConfig
from sequence_model.lstm_crf_model import LSTMCrfModel
from sequence_model.simple_lexicon_model import SimpleLexiconModel

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
    else:
        train = list(zip(train[0], train[1])), train[2]
        test = list(zip(test[0], test[1])), test[2]
        dev = list(zip(dev[0], dev[1])), dev[2]
    return train, test, dev


def create_model():
    tagging_config = SequenceTaggingConfig('lstm-crf')
    model = LSTMCrfModel(tagging_config)
    return model


def train_model():
    model = create_model()
    model.train(train, dev)
    model.evaluate(test)


def predict():
    model = create_model()
    model.restore_session()
    model.evaluate(test)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    train, test, dev = load_data(with_lexicon=False)
    train_model()
    # predict()
