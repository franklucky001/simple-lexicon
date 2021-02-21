# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def mask_predictions(predictions, sequence_lengths):
    masked = []
    for length, prediction in zip(sequence_lengths, predictions):
        masked.append(prediction[:length])
    return masked


def tagging_accuracy_score(labels, predictions):
    flat_labels = np.concatenate(labels, axis=0)
    flat_predictions = np.concatenate(predictions, axis=0)
    return accuracy_score(flat_labels, flat_predictions)


def tagging_f1_score(labels, predictions, average='weighted'):
    flat_labels = np.concatenate(labels, axis=0)
    flat_predictions = np.concatenate(predictions, axis=0)
    assert average in ['micro', 'macro', 'weighted']
    return f1_score(flat_labels, flat_predictions, average=average)


def tagging_precision_score(labels, predictions, average='weighted'):
    flat_labels = np.concatenate(labels, axis=0)
    flat_predictions = np.concatenate(predictions, axis=0)
    assert average in ['micro', 'macro', 'weighted']
    return precision_score(flat_labels, flat_predictions, average=average)


def tagging_recall_score(labels, predictions, average='weighted'):
    flat_labels = np.concatenate(labels, axis=0)
    flat_predictions = np.concatenate(predictions, axis=0)
    assert average in ['micro', 'macro', 'weighted']
    return recall_score(flat_labels, flat_predictions, average=average)