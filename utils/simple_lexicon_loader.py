# -*- coding:utf-8 -*-
import os
import json
from collections import defaultdict
from config.dataset_config import DatasetConfig


class SimpleLexiconLoader:
    LEXICON_UNK = '<UNK>'
    LEXICON_NONE = '<NONE>'
    WORD_UNK = '__unk__'
    WORD_PAD = '__padding__'
    TAG_PAD = 'S-PAD'

    def __init__(self):
        self.use_cache = True
        self.lexicon_score_bias = 1
        self.word2id = {}
        self.id2word = {}
        self.lexicon2id = {}
        self.id2lexicon = {}
        self.lexicon2z_score = {}
        self.lexicon_id2z_score = {}
        self.tag2id = {}
        self.id2tag = {}

    def load_data(self, train_file, test_file, dev_file, lexicon_path):
        if not self.use_cache:
            train_word_set, train_lexicon_set, train_tag_set, lexicon_counter = self.build_lexicon(train_file)
            self.save_word_dic(words=train_word_set, lexicon_path=lexicon_path)
            self.save_lexicon_dic(lexicons=train_lexicon_set, lexicon_path=lexicon_path)
            self.save_tag_dic(tags=train_tag_set, lexicon_path=lexicon_path)
            self.read_word_dic(lexicon_path)
            self.read_lexicon_dic(lexicon_path)
            self.read_tag_dic(lexicon_path)
            self.calc_z_score(lexicon_counter)
            self.save_z_score(lexicon_path)
            train = self.make_samples(train_file)
            test = self.make_samples(test_file)
            dev = self.make_samples(dev_file)
            self.save_samples(train, 'train', lexicon_path)
            self.save_samples(test, 'test', lexicon_path)
            self.save_samples(dev, 'dev', lexicon_path)
        else:
            train = self.load_samples('train', lexicon_path)
            test = self.load_samples('test', lexicon_path)
            dev = self.load_samples('dev', lexicon_path)
            self.read_word_dic(lexicon_path)
            self.read_lexicon_dic(lexicon_path)
            self.read_tag_dic(lexicon_path)
            self.load_z_score(lexicon_path)
        print(f'tag pad id: {self.tag2id[self.TAG_PAD]}')
        print(f'word pad id: {self.word2id[self.WORD_PAD]}')
        print(f'word unk id: {self.word2id[self.WORD_UNK]}')
        print(f'lexicon none id : {self.lexicon2id[self.LEXICON_NONE]}')
        print(f'lexicon unk id : {self.lexicon2id[self.LEXICON_UNK]}')
        return train, test, dev

    @staticmethod
    def save_samples(data, name, lexicon_path):
        filename = f'{name}-record.json'
        filepath = os.path.join(lexicon_path, filename)
        words2id_list, lexicons2id_list, tags2id_list = data
        format_data = {
            'words': words2id_list,
            'lexicons': lexicons2id_list,
            'tags': tags2id_list
        }
        with open(filepath, 'w') as fw:
            fw.write(json.dumps(format_data, indent=2))

    @staticmethod
    def load_samples(name, lexicon_path):
        filename = f'{name}-record.json'
        filepath = os.path.join(lexicon_path, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        words2id_list, lexicons2id_list, tags2id_list = data['words'], data['lexicons'], data['tags']
        return words2id_list, lexicons2id_list, tags2id_list

    def make_samples(self, filepath):
        words2id_list, lexicons2id_list, tags2id_list = [], [], []
        word_unk_id = self.word2id[self.WORD_UNK]
        lexicon_unk_id = self.lexicon2id[self.LEXICON_UNK]
        for words, lexicons, tags in self.reader_sent_gen(filepath):
            words2id = [self.word2id.get(w, word_unk_id) for w in words]
            words2id_list.append(words2id)
            tags2id = [self.tag2id.get(t) for t in tags]
            tags2id_list.append(tags2id)
            lexicons2id = [[self.lexicon2id.get(l, lexicon_unk_id) for l in lexicons[i]] for i in range(4)]
            lexicons2id_list.append(lexicons2id)
        return words2id_list, lexicons2id_list, tags2id_list

    def make_sample_one(self, sample):
        pass

    def calc_z_score(self, lexicon_counter):
        unk_z = min(lexicon_counter.values())
        sum_of_z = sum([lexicon_counter[k] + self.lexicon_score_bias for k in lexicon_counter.keys()])
        score = 1.0 * unk_z / sum_of_z
        self.lexicon2z_score[self.LEXICON_UNK] = score
        for lexicon, count in lexicon_counter.items():
            z = count + self.lexicon_score_bias
            score = 1.0 * z / sum_of_z
            self.lexicon2z_score[lexicon] = score
        for lexicon, score in self.lexicon2z_score.items():
            lexicon_id = self.lexicon2id[lexicon]
            self.lexicon_id2z_score[lexicon_id] = score

    def save_z_score(self, lexicon_path):
        score_file = 'z-score.dic.txt'
        filepath = os.path.join(lexicon_path, score_file)
        with open(filepath, 'w') as fw:
            for lexicon, idx in self.lexicon2id.items():
                score = self.lexicon2z_score[lexicon]
                fw.write(f'{idx}\t{lexicon}\t{score}\n')

    def load_z_score(self, lexicon_path):
        score_file = 'z-score.dic.txt'
        filepath = os.path.join(lexicon_path, score_file)
        with open(filepath, 'r') as f:
            for line in f:
                idx, lexicon, score = line.strip().split('\t')
                score = float(score)
                self.lexicon2z_score[lexicon] = score
                self.lexicon_id2z_score[idx] = score

    @staticmethod
    def reader_sent_gen(filepath):
        with open(filepath, 'r') as f:
            words, lexicons, tags = [], [[], [], [], []], []
            line = f.readline()
            while line:
                if line.strip() == '':
                    yield words, lexicons, tags
                    words, lexicons, tags = [], [[], [], [], []], []
                    line = f.readline()
                    continue
                word, lexicon_str, tag = line.strip().split('\t')
                words.append(word)
                tags.append(tag)
                lexicon_groups = lexicon_str.split('|')
                for i in range(4):
                    group_item_str = lexicon_groups[i]
                    lexicon_group_items = group_item_str.split(',')
                    for lexicon in lexicon_group_items:
                        if lexicon == '':
                            lexicons[i].append(SimpleLexiconLoader.LEXICON_NONE)
                        else:
                            lexicons[i].append(lexicon)
                line = f.readline()

    def build_lexicon(self, filepath):
        word_set, lexicon_set, tag_set = set(), set(), set()
        lexicon_counter = defaultdict(int)
        for words, lexicons, tags in self.reader_sent_gen(filepath):
            word_set |= set(words)
            tag_set |= set(tags)
            for i in range(4):
                lexicon_set |= set(lexicons[i])
                for lexicon in lexicons[i]:
                    lexicon_counter[lexicon] += 1
        return word_set, lexicon_set, tag_set, lexicon_counter

    @staticmethod
    def save_word_dic(words, lexicon_path):
        word_dic_file = os.path.join(lexicon_path, 'word.dic.txt')
        with open(word_dic_file, 'w') as fw:
            fw.write(f'{SimpleLexiconLoader.WORD_PAD}\n')
            fw.write(f'{SimpleLexiconLoader.WORD_UNK}\n')
            for word in words:
                fw.write(word + '\n')

    @staticmethod
    def save_tag_dic(tags, lexicon_path):
        tag_dic_file = os.path.join(lexicon_path, 'tag.dic.txt')
        with open(tag_dic_file, 'w') as fw:
            fw.write(f'{SimpleLexiconLoader.TAG_PAD}\n')
            for tag in tags:
                fw.write(tag + '\n')

    @staticmethod
    def save_lexicon_dic(lexicons, lexicon_path):
        lexicon_dic_file = os.path.join(lexicon_path, 'lexicon.dic.txt')
        with open(lexicon_dic_file, 'w') as fw:
            fw.write(f'{SimpleLexiconLoader.LEXICON_UNK}\n')
            for lexicon in lexicons:
                fw.write(lexicon + '\n')

    def read_word_dic(self, lexicon_path):
        word_dic_file = os.path.join(lexicon_path, 'word.dic.txt')
        with open(word_dic_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                word = line.strip()
                self.word2id[word] = i
                self.id2word[i] = word

    def read_lexicon_dic(self, lexicon_path):
        lexicon_dic_file = os.path.join(lexicon_path, 'lexicon.dic.txt')
        with open(lexicon_dic_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                lexicon = line.strip()
                self.lexicon2id[lexicon] = i
                self.id2lexicon[i] = lexicon

    def read_tag_dic(self, lexicon_path):
        tag_dic_file = os.path.join(lexicon_path, 'tag.dic.txt')
        with open(tag_dic_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                tag = line.strip()
                self.tag2id[tag] = i
                self.id2tag[i] = tag


if __name__ == "__main__":
    config = DatasetConfig(version='v1')
    loader = SimpleLexiconLoader()
    train_data, test_data, dev_data = loader.load_data(config.train_file, config.test_file, config.dev_file, config.lexicon_path)
    print('success')
