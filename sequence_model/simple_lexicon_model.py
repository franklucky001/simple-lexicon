# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from utils.data_utils import pad_sequences, pad_char_level_sequences
from sequence_model.base_sequence_tagging import BaseSequenceTagging


class SimpleLexiconModel(BaseSequenceTagging):
    def __init__(self, config):
        self.b_lexicon_ids = None
        self.m_lexicon_ids = None
        self.e_lexicon_ids = None
        self.s_lexicon_ids = None
        self.word_lengths = None
        super().__init__(config)

    def add_placeholders(self):
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        # shape = (batch size, max length of sentence, max length of lexicon)
        self.b_lexicon_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='begin_lexicon_ids')
        self.m_lexicon_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='middle_lexicon_ids')
        self.e_lexicon_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='end_lexicon_ids')
        self.s_lexicon_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='single_lexicon_ids')
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

        # shape = (batch size, max length of sentence in batch)
        self.predictions = tf.placeholder(tf.int32, shape=[None, None], name="predictions")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def add_word_embeddings(self):
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.n_words, self.config.word_dim])
            else:
                _word_embeddings = tf.Variable(
                    self.config.embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids, name="word_embeddings")

        with tf.variable_scope("lexicons"):
            _lexicon_embeddings = tf.get_variable(
                name='_lexicon_embeddings',
                dtype=tf.float32,
                shape=[self.config.n_lexicons, self.config.lexicon_dim]
            )
            _lexicon_scores = tf.Variable(
                self.config.lexicon_z_embeddings,
                name="_lexicon_score",
                dtype=tf.float32,
                trainable=False
            )
            lexicon_embeddings_list = []
            all_lexicon_ids = [self.b_lexicon_ids, self.m_lexicon_ids, self.e_lexicon_ids, self.s_lexicon_ids]
            suffix_list = ['begin', 'middle', 'end', 'single']
            for lexicon_ids, suffix in zip(all_lexicon_ids, suffix_list):
                lexicon_embeddings = self.add_lexicon_mul_score(lexicon_ids, suffix, _lexicon_embeddings, _lexicon_scores)
                lexicon_embeddings = tf.reduce_sum(lexicon_embeddings, axis=2)
                lexicon_embeddings_list.append(lexicon_embeddings)
            output = tf.concat(lexicon_embeddings_list, axis=-1)
            word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    @staticmethod
    def add_lexicon_mul_score(lexicon_ids, suffix, _lexicon_embeddings, _lexicon_scores):
        embedding_lexicon = tf.nn.embedding_lookup(_lexicon_embeddings,
                                                   lexicon_ids,
                                                   name=f'lexicon_embedding_{suffix}')
        lexicon_score = tf.nn.embedding_lookup(_lexicon_scores,
                                               lexicon_ids,
                                               name=f'lexicon_score_{suffix}')
        embedding_output = lexicon_score * embedding_lexicon
        return embedding_output

    def add_logits_layer(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("project"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[2*self.config.hidden_size_lstm, self.config.n_tags])

            b = tf.get_variable("b", shape=[self.config.n_tags], dtype=tf.float32, initializer=tf.zeros_initializer())

            n_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, n_steps, self.config.n_tags])

    def get_feed_dict(self, x_batch, y_batch=None, lr=None, dropout=None):
        words_batch = [item[0] for item in x_batch]
        lexicons_batch = [[item[1][k] for item in x_batch] for k in range(4)]
        max_length = max(map(lambda x: len(x), words_batch))
        word_ids, sequence_lengths = pad_sequences(words_batch, 0, max_length)
        lexicon_ids_list = []
        for i in range(4):
            lexicon_ids, _ = pad_char_level_sequences(lexicons_batch[i], 0)
            lexicon_ids = np.array(lexicon_ids)
            lexicon_ids_list.append(lexicon_ids)
        feed = {
            self.word_ids: word_ids,
            self.b_lexicon_ids: lexicon_ids_list[0],
            self.m_lexicon_ids: lexicon_ids_list[1],
            self.e_lexicon_ids: lexicon_ids_list[2],
            self.s_lexicon_ids: lexicon_ids_list[3],
            self.sequence_lengths: sequence_lengths
        }
        if y_batch is not None:
            labels, _ = pad_sequences(y_batch, 0, max_length)
            feed[self.labels] = labels
        if lr is not None:
            feed[self.lr] = lr
        if dropout is not None:
            feed[self.dropout] = dropout
        return feed, sequence_lengths

    def predict(self, x):
        # TODO override
        pass