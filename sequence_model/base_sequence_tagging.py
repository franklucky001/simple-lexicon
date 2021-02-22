# -*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from utils.tagging_metrics import tagging_accuracy_score, tagging_f1_score, tagging_precision_score, mask_predictions
from config.sequence_tagging_config import SequenceTaggingConfig
from utils.data_utils import pad_sequences


class BaseSequenceTagging:

    def __init__(self, config: SequenceTaggingConfig):
        self.config = config
        self.session: tf.Session = None
        self.saver = None
        self.merged = None
        self.summary_writer = None
        self.word_ids = None
        self.word_embeddings = None
        self.sequence_lengths = None
        self.labels = None
        self.predictions = None
        self.logits = None
        self.trans_params = None
        self.likelihood = None
        self.lr = None
        self.dropout = None
        self.losses = None
        self.train_op = None
        self.logger = config.logger
        self.build()

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings()
        self.add_logits_layer()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_train_op()
        self.initialize_session()

    def add_placeholders(self):
        pass

    def add_word_embeddings(self):
        pass

    def add_logits_layer(self):
        pass

    def add_prediction_op(self):
        if self.config.use_crf:
            with tf.variable_scope("prediction-crf"):
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
                self.likelihood = log_likelihood
                self.trans_params = trans_params  # need to evaluate it for decodin
                viterbi_seq, viterbi_score = tf.contrib.crf.crf_decode(self.logits,
                                                                       self.trans_params,
                                                                       self.sequence_lengths)
                self.predictions = viterbi_seq
        else:
            with tf.variable_scope("prediction"):
                self.predictions = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        with tf.variable_scope("losses"):
            if self.config.use_crf:
                self.losses = tf.reduce_mean(-self.likelihood)
            else:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(loss, mask)
                self.losses = tf.reduce_mean(losses)

    def add_train_op(self):
        optimizer_method = self.config.optimizer.lower()
        with tf.variable_scope('train_op'):
            if optimizer_method == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif optimizer_method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif optimizer_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer_method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise NotImplementedError("Unknown method {}".format(optimizer_method))
            '''clip grad'''
            if self.config.clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(self.losses))
                grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.losses)

    def train(self, train, dev):
        if self.config.do_summary:
            self.add_summary()
        ckpt = os.path.join(self.config.model_dir, '.index')
        if os.path.exists(ckpt):
            self.logger.info("model is exists, restore form cache")
            self.restore_session()
        else:
            self.logger.info("first training which no cache")
        best_score = 0.0
        epochs_no_improve = 0
        for epoch in range(self.config.epochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.epochs))
            score = self.run_epoch(train, dev, epoch, self.config.do_summary)
            self.config.lr *= self.config.lr_decay
            if score > best_score:
                epochs_no_improve = 0
                self.save_session()
                best_score = score
                self.logger.info(f"- new best score {score:06.4f}!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.epochs_no_improve:
                    self.logger.info(f"- early stopping {epochs_no_improve} epochs without improve")
                    break

    def run_epoch(self, train, dev, epoch, do_summary):
        n_batch = 0
        x_train, y_train = train
        n_batches = (len(y_train) + self.config.batch_size - 1) // self.config.batch_size
        for x_batch, y_batch in self.mini_batches(x_train, y_train):
            feed, sequence_lengths = self.get_feed_dict(x_batch, y_batch, self.config.lr, self.config.dropout)
            if do_summary:
                fetches = [self.train_op, self.losses, self.predictions, self.merged]
                _, train_loss, prediction, summary = self.session.run(fetches, feed_dict=feed)
                if n_batch % 100 == 0:
                    self.summary_writer.add_summary(summary, epoch * n_batches + n_batch)
                n_batch += 1
            else:
                fetches = [self.train_op, self.losses, self.predictions]
                _, train_loss, prediction = self.session.run(fetches, feed_dict=feed)
                if n_batch % 100 == 0:
                    self.logger.info(f"batch {n_batch} of batches {n_batches},training loss:{train_loss:04.2f}")
                n_batch += 1
        metrics = self.run_evaluate(dev)
        msg = " - ".join([f'Epoch {epoch+1} dev score {k} {v:04.2f}' for k, v in metrics.items()])
        self.logger.info(msg)
        return metrics['f1']

    def evaluate(self, test):
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join([f"{k} {v:06.4f}" for k,v in metrics.items()])
        self.logger.info(msg)

    def run_evaluate(self, test):
        features, labels = test
        predictions = []
        for x_batch, y_batch in self.mini_batches(features, labels):
            pred_batch, batch_sequence_lengths = self.predict_batch(x_batch)
            prediction = mask_predictions(pred_batch, sequence_lengths=batch_sequence_lengths)
            predictions.extend(prediction)
        acc = tagging_accuracy_score(labels, predictions)
        precision = tagging_precision_score(labels, predictions, average='macro')
        f1 = tagging_f1_score(labels, predictions, average='macro')
        return {'accuracy': acc, 'precision': precision, 'f1': f1}

    def predict_batch(self, x_batch):
        feed, sequence_lengths = self.get_feed_dict(x_batch, dropout=1.0)
        predictions = self.session.run(self.predictions, feed_dict=feed)
        return predictions, sequence_lengths

    def predict(self, x):
        x_batch = np.expand_dims(x, axis=0)
        feed = self.get_feed_dict(x_batch)
        pred_batch = self.session.run(self.labels, feed_dict=feed)
        return pred_batch[0]

    def initialize_session(self):
        self.logger.info("init tf session")
        if self.config.use_gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = None
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_session(self):
        self.logger.info("restore session")
        self.saver.restore(self.session, self.config.model_dir)

    def save_session(self):
        self.logger.info("save session")
        self.saver.save(self.session, self.config.model_dir)

    def close_session(self):
        self.session.close()

    def add_summary(self):
        summary_inputs = []
        with tf.Graph().as_default():
            with tf.variable_scope('losses'):
                summary_loss = tf.summary.scalar('losses', self.losses)
                summary_inputs.append(summary_loss)
        self.merged = tf.summary.merge(summary_inputs)
        self.summary_writer = tf.summary.FileWriter(self.config.summary_dir, self.session.graph)

    def mini_batches(self, features, labels):
        n_samples = len(labels)
        batch_size = self.config.batch_size
        offset = 0
        while offset < n_samples:
            x, y = features[offset:offset+batch_size], labels[offset:offset+batch_size]
            yield x, y
            offset += batch_size

    def get_feed_dict(self, x_batch, y_batch=None, lr=None, dropout=None):
        max_length = max(map(lambda x: len(x), x_batch))
        word_ids, sequence_lengths = pad_sequences(x_batch, 0, max_length)
        feed = {
            self.word_ids: word_ids,
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
