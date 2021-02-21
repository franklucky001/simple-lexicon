# -*- coding:utf-8 -*-
import jieba
import pkuseg
pku_tokenizer = pkuseg.pkuseg()


def make_simple_lexicon_data(ori_file, dst_file):
    fw = open(dst_file, 'w')
    for words, tags in read_sent_gen(ori_file):
        sent = ''.join(words)
        ranges_all = make_term_ranges_by_segment(sent)
        lexicon_groups = []
        offset = 0
        for word in words:
            lexicon_group = make_lexicon_group(offset, ranges_all)
            lexicon_groups.append(lexicon_group)
            offset += len(word)
        for word, lexicon_group, tag in zip(words, lexicon_groups, tags):
            fw.write(f'{word}\t{lexicon_group}\t{tag}\n')
        fw.write('\n')
    fw.close()


def make_term_ranges_by_segment(sent):
    terms_1 = jieba.lcut(sent, cut_all=True)
    ranges_1 = make_range(sent, terms_1)
    terms_2 = pku_tokenizer.cut(sent)
    ranges_2 = make_range(sent, terms_2)
    ranges_all = list(set(ranges_1) | set(ranges_2))
    ranges_all.sort(key=lambda item: item[1])
    return ranges_all


def make_lexicon_group(offset, ranges):
    group_b = filter(lambda item: item[1] == offset and item[2] > offset+1, ranges)
    group_m = filter(lambda item: item[1] < offset and item[2] > offset+1, ranges)
    group_e = filter(lambda item: item[1] < offset and item[2] == offset+1, ranges)
    group_s = filter(lambda item: item[1] == offset and item[2] == offset+1, ranges)
    words_b = [g[0] for g in group_b]
    words_m = [g[0] for g in group_m]
    words_e = [g[0] for g in group_e]
    words_s = [g[0] for g in group_s]
    lexicon_group = '|'.join([','.join(words_g) for words_g in [words_b, words_m, words_e, words_s]])
    return lexicon_group


def make_range(sent, words):
    ranges = []
    offset = 0
    for word in words:
        pos = sent[offset:].find(word)
        beg = offset + pos
        end = offset + pos + len(word)
        if ranges:
            prev = ranges[-1]
            if prev[2] < beg:
                offset = prev[2]
        ranges.append((word, beg, end))
    return ranges


def read_sent_gen(ori_file):
    with open(ori_file, 'r') as f:
        line = f.readline()
        words, tags = [], []
        while line:
            line = line.strip()
            if line == '':
                yield words, tags
                words, tags = [], []
                line = f.readline()
                continue
            word, tag = line.split(' ')
            words.append(word)
            tags.append(tag)
            line = f.readline()
