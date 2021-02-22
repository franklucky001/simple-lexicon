# -*- coding:utf-8 -*-
import os
from config.dataset_config import DatasetConfig
from utils.simple_lexicon_loader import SimpleLexiconLoader
from config.sequence_tagging_config import SequenceTaggingConfig
from sequence_model.lstm_crf_model import LSTMCrfModel
from sequence_model.simple_lexicon_model import SimpleLexiconModel


def load_data(with_lexicon=False):
    loader = SimpleLexiconLoader(use_cache=True)
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
    if use_lexicon_model:
        model = SimpleLexiconModel(tagging_config)
        print("use simple lexicon model")
    else:
        model = LSTMCrfModel(tagging_config)
        print("use basic lstm crf model")
    return model


def train_model():
    global model
    model.train(train, dev)
    # model.evaluate(test)


def predict():
    global model
    model.restore_session()
    model.evaluate(test)


if __name__ == "__main__":
    use_lexicon = os.environ.get('USE_LEXICON', 'False')
    use_lexicon_model = True if use_lexicon.lower() == 'true' else False
    with_lexicon = use_lexicon_model
    model_name = os.environ.get('MODEL_NAME', 'lstm-crf')
    data_name = os.environ.get('DATA_NAME', 'rmrb')
    config = DatasetConfig(data_name=data_name)
    tagging_config = SequenceTaggingConfig(model_name, data_name)
    train, test, dev = load_data(with_lexicon)
    model = create_model()
    # train_model()
    predict()
