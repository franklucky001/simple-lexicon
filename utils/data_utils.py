# -*- coding:utf-8 -*-
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_char_level_sequences(sequences, pad_tok):
    max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        # all words are same length now
        sp, sl = pad_sequences(seq, pad_tok, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]
    max_length_sentence = max(map(lambda x: len(x), sequences))
    sequence_padded, _ = pad_sequences(sequence_padded,
                                        [pad_tok] * max_length_word, max_length_sentence)
    sequence_length, _ = pad_sequences(sequence_length, 0,
                                        max_length_sentence)
    return sequence_padded, sequence_length

def mini_batches(data, mini_batch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        mini_batch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == mini_batch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def load_z_vectors(filepath):
    import numpy as np
    scores = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            _, _, score = line.strip().split()
            scores.append([score])
    return np.array(scores)
