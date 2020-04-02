import collections
import itertools
import operator
import copy
import numpy as np
import tensorflow as tf

class NERData(object):
    def __init__(self, data_fname, char2id=None, tag2id=None):
        """
        将所有的句子组成一个列表，统计每个字符的频数，构建字典
        将所有的标签构建一个字典
        将所有句子按照最大长度padding，转为id，存入列表中
        """
        sentences = self._load_data(data_fname)

        chars, tags = zip(*itertools.chain(*sentences))
        if char2id is None:

            self._m_char2id = self._build_dict(chars)
        else:
            self._m_char2id = copy.deepcopy(char2id)

        if tag2id is None:
            self._m_tag2id = {tag: i for i, tag in enumerate(sorted(set(tags)))}
        else:
            self._m_tag2id = copy.deepcopy(tag2id)
        self._m_sentences = self._convert_sentence_to_ids(sentences)

    @property
    def char2id(self):
        return self._m_char2id

    @property
    def id2char(self):
        return {i: c for c, i in self._m_char2id.items()}

    @property
    def tag2id(self):
        return self._m_tag2id

    @property
    def id2tag(self):
        return {i: t for t, i in self._m_tag2id.items()}

    def _load_data(self, fname):
        # 将所有句子组成一个列表
        sentences = list()
        with open(fname, 'r',encoding='utf-8') as fin:
            sentence = list()
            for line in fin:
                fields = line.strip().split()
                if len(fields) != 2:
                    sentences.append(sentence)
                    sentence = list()
                else:
                    sentence.append(fields)
            if len(sentence) > 0:
                sentences.append(sentence)
        return sentences

    def _build_dict(self, chars):
        char_cnt = collections.Counter(chars)
        char_list, _ = zip(*sorted(char_cnt.items(), key=operator.itemgetter(1), reverse=True))
        return {w: i for i, w in enumerate(["<PAD>", "<UNK>"] + list(char_list))}

    def _convert_sentence_to_ids(self, sentences):
        unknown_char_id = self._m_char2id["<UNK>"]
        padding_char_id = self._m_char2id["<PAD>"]

        char_ids = [[self._m_char2id.get(c, unknown_char_id) for c, t in s] for s in sentences]
        tag_ids = [[self._m_tag2id[t] for c, t in s] for s in sentences]

        sequence_lengths = [len(s) for s in char_ids]
        max_length = max(sequence_lengths)

        char_ids = [s + [padding_char_id] * (max_length - len(s)) for s in char_ids]
        tag_ids = [s + [self._m_tag2id["O"]] * (max_length - len(s)) for s in tag_ids]

        return list(map(np.array, [sequence_lengths, char_ids, tag_ids]))

    def iter_batches(self, batch_size=16, shuffle=False):
        sentence_lengths, char_ids, tag_ids = self._m_sentences
        sentence_idx = np.arange(sentence_lengths.shape[0])
        if shuffle is True:
            np.random.shuffle(sentence_idx)

        for i in range(0, sentence_lengths.shape[0], batch_size):
            batch_idx = sentence_idx[i: i + batch_size]
            yield sentence_lengths[batch_idx], char_ids[batch_idx], tag_ids[batch_idx]

