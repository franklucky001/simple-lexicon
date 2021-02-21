# -*- coding:utf-8 -*-
import os
import logging
from utils.data_utils import load_vocab


class SequenceTaggingConfig:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, model_name):
        self.word_dim = 300
        self.lexicon_dim = 100
        self.char_dim = 100
        self.hidden_size_lstm = 300
        self.hidden_size_char = 100
        self.use_chars = False
        self.use_crf = True
        self.train_embeddings = False
        self.epochs = 10
        self.epochs_no_improve = 3
        self.lr = 1e-3
        self.lr_decay = 0.9
        self.dropout = 0.5
        self.clip = 5.0
        self.batch_size = 64
        self.do_summary = True
        self.optimizer = 'adam'
        self.model_dir = f'{self.base_dir}/models/{model_name}/'
        self.summary_dir = f'{self.base_dir}/summary/{model_name}/'
        self.logger_path = f'{self.base_dir}/logs/{model_name}-log.txt'
        self.dataset_path = os.path.join(self.base_dir, 'data/records/resume')
        self.words_file = os.path.join(self.dataset_path, 'word.dic.txt')
        self.tags_file = os.path.join(self.dataset_path, 'tag.dic.txt')
        self.chars_file = ''
        self.lexicons_file = os.path.join(self.dataset_path, 'lexicon.dic.txt')
        self.logger = None
        self.embeddings = None
        self.vocab_words = dict()
        self.vocab_tags = dict()
        self.vocab_chars = dict()
        self.vocab_lexicons = dict()
        self.init()
        self.n_words = len(self.vocab_words)
        self.n_chars = len(self.vocab_chars)
        self.n_tags = len(self.vocab_tags)
        self.n_lexicons = len(self.vocab_lexicons)

    def init(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.logger = self.get_logger(self.logger_path)
        self.vocab_words = load_vocab(self.words_file)
        self.vocab_tags = load_vocab(self.tags_file)
        self.vocab_lexicons = load_vocab(self.lexicons_file)
        if self.use_chars:
            self.vocab_chars = load_vocab(self.chars_file)
        else:
            self.vocab_chars = {}

    @staticmethod
    def get_logger(filename):
        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        return logger