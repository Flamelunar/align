import numpy as np
from common import *


class Instance(object):
    def __init__(self, id, lines, domain_id):
        self.id = id
        n1 = len(lines) + 1
        self.words_s = [''] * n1
        self.tags_s = [''] * n1
        self.heads_s = [''] * n1
        self.labels_s = [''] * n1
        self.labels_s_predict = [''] * n1
        self.words_i = np.array([-1] * n1, dtype=data_type_int)
        self.ext_words_i = np.array([-1] * n1, dtype=data_type_int)
        self.tags_i = np.array([-1] * n1, dtype=data_type_int)
        self.heads_i = np.array([-1] * n1, dtype=data_type_int)
        self.labels_i = np.array([-1] * n1, dtype=data_type_int)
        self.heads_i_predict = np.array([-1] * n1, dtype=data_type_int)
        self.labels_i_predict = np.array([-1] * n1, dtype=data_type_int)
        self.chars_i = np.zeros((n1, 39), dtype=data_type_int)  # max_char=32
        self.word_lens = np.array([1] * n1, dtype=data_type_int)
        self.words_s[0] = pseudo_word_str
        self.tags_s[0] = pseudo_word_str
        # self.heads_s[0] = str(ignore_id_head_or_label)
        self.heads_i[0] = ignore_id_head_or_label
        self.labels_i[0] = ignore_id_head_or_label
        self.heads_i_predict[0] = ignore_id_head_or_label
        self.labels_i_predict[0] = ignore_id_head_or_label
        self.lstm_mask = None

        self.domains_nadv_i = np.array([domain_id] * n1, dtype=data_type_int)
        self.domain_id_nadv = domain_id
        self.domains_nadv_i[0] = ignore_domain
        # if domain_id !=2:
        #    domain_id = 1
        # self.domains_i = np.array([domain_id] * n1, dtype=data_type_int)
        # self.domain_id = domain_id
        # self.domains_i = np.array([domain_id] * n1, dtype=data_type_int)
        # domains=np.array([1,2])
        # if domain_id == 2:
        #    self.domains_i = np.random.choice(a=domains, size=n1, replace=True, p=[0.2,0.8])#yli:19-5-5
        # else:
        #    self.domains_i = np.random.choice(a=domains, size=n1, replace=True, p=[0.8,0.2])#yli:19-5-5
        # self.domains_i[0] = ignore_domain
        # self.domain_id =domain_id

        domains = np.array([1, 2])
        if domain_id == 2:
            self.domains_i = np.random.choice(a=domains, size=n1, replace=True, p=[0.2, 0.8])  # yli:19-5-5
        else:
            self.domains_i = np.random.choice(a=domains, size=n1, replace=True, p=[0.8, 0.2])  # yli:19-5-5
        self.domains_i[0] = ignore_domain

        self.decompose_sent(lines)

    def size(self):
        return len(self.words_s)

    def word_num(self):
        return self.size() - 1

    @staticmethod
    def compose_sent(words_s, tags_s, heads_i, labels_s):
        n1 = len(words_s)
        assert ((n1,) * 3 == (len(tags_s), len(heads_i), len(labels_s)))
        lines = [''] * (n1 - 1)
        for i in np.arange(1, n1):
            lines[i - 1] = ("%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n" %
                            (i, words_s[i], "_",
                             tags_s[i], "_", "_",
                             heads_i[i], labels_s[i],
                             "_", "_"))
        return lines

    def write(self, out_file):
        lines = Instance.compose_sent(self.words_s, self.tags_s, self.heads_i_predict, self.labels_s_predict)
        for line in lines:
            out_file.write(line)
        out_file.write('\n')

    def decompose_sent(self, lines):
        """
        :param lines: 它的主要功能是将文本行 lines 解析并从中提取信息，然后将这些信息存储在不同的属性中。
        :return:
        """
        for (idx, line) in enumerate(lines):
            i = idx + 1  # 索引从1开始而不是0
            tokens = line.strip().split('\t')
            # print(tokens)
            # print(tokens[1], tokens[3], tokens[6], tokens[7],)
            """
            0. 序号（Index）：在句子中的单词位置，从 1 开始递增。但是列表token从0开始计数
            1. 单词形式（Form）：单词的原始形式。
            2. Lemma（或词干）：单词的基本形式。
            3. 词性（Part of Speech）：单词的词性或词类。  POS任务词性标签
            4. Universal POS（通用词性标记）：通用词性标记，通常是针对多种语言的标记体系。
            5. 附加信息（Additional Information）：其他信息，如语法特征。
            6. 依存关系头部（Head）：与当前单词存在依存关系的单词的序号。
            7. 依存关系标签（Dependency Label）：描述与当前单词之间的依存关系的标签。
            8. 其他字段（Additional Fields）：可能包括其他语法或语义信息。
            tokens[1]` 包含一个单词的文本形式。   tokens[3]` 包含一个单词的词性标签。
            tokens[6]` 包含了与当前单词存在依存关系的单词的索引或标识。
            tokens[7]` 包含了描述依存关系的标签。
            """
            assert (len(tokens) >= 8)
            self.words_s[i], self.tags_s[i], self.heads_s[i], self.labels_s[i] = \
                tokens[1], tokens[3], tokens[6], tokens[7]
            self.heads_i[i] = int(self.heads_s[i])
            if self.heads_i[i] < 0:
                assert self.heads_i[i] == ignore_id_head_or_label

    def eval(self):
        word_num_to_eval = 0
        word_num_arc_correct = 0
        word_num_label_correct = 0
        for i in np.arange(1, self.size()):
            if self.heads_i[i] < 0:
                continue
            word_num_to_eval += 1
            if self.heads_i[i] != self.heads_i_predict[i]:
                continue
            word_num_arc_correct += 1
            if self.labels_i[i] == self.labels_i_predict[i]:
                word_num_label_correct += 1
        return word_num_to_eval, word_num_arc_correct, word_num_label_correct
