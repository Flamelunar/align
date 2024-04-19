import numpy as np
from common import *


class UnInstance(object):
    def __init__(self, id, lines, domain_id):
        self.id = id
        n1 = len(lines) + 1
        self.words_s = [''] * n1
        self.tags_s = [''] * n1
        self.tags_s_predict = [''] * n1
        self.words_i = np.array([-1] * n1, dtype=data_type_int)
        self.ext_words_i = np.array([-1] * n1, dtype=data_type_int)
        self.tags_i = np.array([-1] * n1, dtype=data_type_int)
        self.chars_i = np.zeros((n1, 39),dtype=data_type_int)#max_char=32
        self.word_lens = np.array([1] * n1, dtype=data_type_int)
        self.words_s[0] = pseudo_word_str
        self.lstm_mask = None
        
        self.domains_nadv_i = np.array([domain_id] * n1, dtype=data_type_int)
        self.domain_id_nadv = domain_id
        self.domains_nadv_i[0] = ignore_domain 
        if domain_id !=4:
            domain_id = 1
        else:
            domain_id = 2
        self.domains_i = np.array([domain_id] * n1, dtype=data_type_int)
        self.domain_id = domain_id
        #self.domains_i = np.array([domain_id] * n1, dtype=data_type_int)
        #domains=np.array([1,2])
        #self.domains_i = np.random.choice(a=domains, size=n1, replace=True, p=[0.5,0.5])#yli:19-4-29
        self.domains_i[0] = ignore_domain 

        #domains=np.array([1,2])
        #self.domains_i = np.random.choice(a=domains, size=n1, replace=True, p=[0.5,0.5])#yli:19-4-29
        #self.domains_i[0] = ignore_domain 

        self.decompose_sent(lines)

    def size(self):
        return len(self.words_s)

    def word_num(self):
        return self.size() - 1

    @staticmethod
    def compose_sent(words_s, tags_s, heads_i, labels_s):
        n1 = len(words_s)
        assert((n1,)*3 == (len(tags_s), len(heads_i), len(labels_s)))
        lines = [''] * (n1 - 1)
        for i in np.arange(1, n1):
            lines[i-1] = ("%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n" %
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
        for (idx, line) in enumerate(lines):
            i = idx + 1
            tokens = line.strip().split('\t')
            assert(len(tokens) >= 8)
            self.words_s[i], self.tags_s[i] = tokens[1], tokens[3]


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


