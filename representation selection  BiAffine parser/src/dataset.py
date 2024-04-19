from vocab import *
import numpy as np
import torch
from torch.autograd import Variable
from k_means import KMeans
from k_means import KMeans
from simple_bucketing import Bucketing
from instance import *
from uninstance import *
import math


class Dataset(object):
    def __init__(self, file_name, max_bucket_num=80, word_num_one_batch=5000, sent_num_one_batch=200,
                 inst_num_max=-1, min_len=1, max_len=100):
        self._file_name = file_name
        self._file_name_short = file_name[-30:].replace('/', '_')

        self._instances = []
        self._batch_num = 0

        print("self._file_name_short", self._file_name_short)
        self._domain_id = 0  # yli add 2019-3-27
        if "ch" in self._file_name_short:
            self._domain_id = 1
            print("self._domain_id", self._domain_id)
        elif "vi" in self._file_name_short:
            self._domain_id = 2
            print("self._domain_id", self._domain_id)
        elif "content" in self._file_name_short:        # 评论数据集
            self._domain_id = 3
            print("self._domain_id", self._domain_id)
        elif "zx" in self._file_name_short:
            self._domain_id = 4
            print("self._domain_id", self._domain_id)

        else:
            print("the filename is wrong, we cann't distinguish its domain", self._file_name_short)

        word_num_total = 0
        with open(self._file_name, mode='r', encoding='utf-8') as f:
            lines = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    length = len(lines)
                    if length >= min_len and (max_len < 0 or length <= max_len):
                        if "unlabel" in self._file_name_short:
                            self._instances.append(
                                UnInstance(len(self._instances), lines, self._domain_id))  # yli 19-5-16
                            # if (inst_num_max > 0) and (self.size() == 16339):#inst_num_max):
                            if self.size() == 21265:  # 16339:#inst_num_max):
                                break
                        else:
                            self._instances.append(Instance(len(self._instances), lines, self._domain_id))
                        word_num_total += length
                        if (inst_num_max > 0) and (self.size() == inst_num_max):
                            break
                    lines = []
                else:
                    lines.append(line)
        assert self.size() > 0
        print('Reading %s done: %d instances' % (self._file_name_short, self.size()), flush=True)

        self.max_len = 0
        self.word_num_accum_so_far = 0

        self._idx_to_read_next_batch = 0
        self._word_num_one_batch = word_num_one_batch
        self._sent_num_one_batch = sent_num_one_batch
        assert self._word_num_one_batch > 0 or self._sent_num_one_batch > 0

        self._bucket_num = -1
        self._use_bucket = (max_bucket_num > 1)
        self._buckets = None  # [(max_len, inst_num_one_batch, bucket)]
        self._bucket_idx_to_read_next_batch = 0

        if self._use_bucket:
            assert (self._word_num_one_batch > 0)
            len_counter = Counter()
            for inst in self.all_inst:
                len_counter[inst.size()] += 1

            # Automatically decide the bucket num according to the data
            self._bucket_num = int(min(max_bucket_num, math.ceil(len(len_counter) / 1.5),
                                       np.ceil(word_num_total / (2 * self._word_num_one_batch))))
            assert self._bucket_num > 0
            # print(self._bucket_num, word_num_total, len(len_counter))

            # k_classes = KMeans(self._bucket_num, len_counter)
            k_classes = Bucketing(self._bucket_num, len_counter)
            max_len_buckets = k_classes.max_len_in_buckets
            len2bucket_idx = k_classes.len2bucket_idx
            self._bucket_num = len(max_len_buckets)
            buckets = [None] * self._bucket_num
            # Can NOT use [[]] * self._bucket_num, shallow copy issue!
            for inst in self.all_inst:
                b_idx = len2bucket_idx[inst.size()]
                if buckets[b_idx] is None:
                    buckets[b_idx] = [inst]
                else:
                    buckets[b_idx].append(inst)
            batch_num_total = 0
            inst_num_one_batch_buckets = []
            for (i, max_len) in enumerate(max_len_buckets):
                inst_num = len(buckets[i])
                batch_num_to_provide = max(1, round(float(inst_num) * max_len / self._word_num_one_batch),
                                           math.ceil(float(inst_num) / self._sent_num_one_batch))
                print("i, inst_num, max_len, batch_num_to_provide, batch_num_total = ", i, inst_num, max_len,
                      batch_num_to_provide, batch_num_total)
                batch_num_total += batch_num_to_provide
                inst_num_one_batch_this_bucket = math.ceil(inst_num / batch_num_to_provide)
                # The goal is to avoid the last batch of one bucket contains too few instances
                inst_num_one_batch_buckets.append(inst_num_one_batch_this_bucket)
                # assert inst_num_one_batch_this_bucket * (batch_num_to_provide-0.5) < inst_num
            print('%s can provide %d batches in total with %d buckets' %
                  (self._file_name_short, batch_num_total, self._bucket_num), flush=True)
            self._batch_num = batch_num_total
            self._buckets = [(ml, nb, b) for ml, nb, b in zip(max_len_buckets, inst_num_one_batch_buckets, buckets)]

    @property
    def file_name_short(self):
        return self._file_name_short

    def size(self):
        return len(self._instances)

    @property
    def batch_num(self):
        return self._batch_num

    @property
    def domain_id(self):
        return self._domain_id

    def _shuffle(self):
        if self._use_bucket:
            for (max_len, inst_num, bucket) in self._buckets:
                np.random.shuffle(bucket)
            np.random.shuffle(self._buckets)
        else:
            np.random.shuffle(self._instances)

    @property
    def all_inst(self):
        return self._instances

    @property
    def all_buckets(self):
        return self._buckets

    def get_one_batch_bucket(self, rewind):
        if self._bucket_idx_to_read_next_batch >= self._bucket_num:
            self._bucket_idx_to_read_next_batch = 0
            assert 0 == self._idx_to_read_next_batch
            if rewind:
                self._shuffle()
            else:
                return

        max_len, inst_num_one_batch, this_bucket = self._buckets[self._bucket_idx_to_read_next_batch]
        inst_num = len(this_bucket)
        assert inst_num > 0
        assert self._idx_to_read_next_batch < inst_num
        inst_num_left = inst_num - self._idx_to_read_next_batch
        inst_num_for_this_batch = min(inst_num_left, inst_num_one_batch)
        idx_next_batch = self._idx_to_read_next_batch + inst_num_for_this_batch
        self.one_batch = this_bucket[self._idx_to_read_next_batch:idx_next_batch]
        assert len(self.one_batch) > 0
        self.max_len = max_len
        for inst in self.one_batch:
            self.word_num_accum_so_far += inst.size()
        if idx_next_batch >= inst_num:
            assert idx_next_batch == inst_num
            self._bucket_idx_to_read_next_batch += 1
            self._idx_to_read_next_batch = 0
        else:
            self._idx_to_read_next_batch = idx_next_batch

    # When all instances are (nearly) consumed, automatically _shuffle
    #   and be ready for the next batch (user transparent).
    # DO NOT USE indices. USE instance directly instead.
    def get_one_batch(self, rewind=True):
        self.one_batch = []
        self.max_len = 0  # include w0
        self.word_num_accum_so_far = 0
        to_return = False
        if self._use_bucket:
            self.get_one_batch_bucket(rewind)
            to_return = True
        else:
            inst_num_left = self.size() - self._idx_to_read_next_batch
            # assume 25 is the averaged #token in a sentence
            # if inst_num_left <= 0 or \
            #         (word_num is not None) and (inst_num_left * 25 < word_num / 2) \
            #         or sent_num < inst_num_left / 2:
            # The above is a more complex way
            # The following way: a batch can consist of only one instance
            if inst_num_left <= 0:
                if rewind:
                    self._shuffle()
                else:
                    to_return = True
                self._idx_to_read_next_batch = 0

        while to_return is False:
            sz = self._instances[self._idx_to_read_next_batch].size()  # include w0
            if (self._word_num_one_batch > 0) and \
                    (self.word_num_accum_so_far + sz > self._word_num_one_batch + 25):
                break  # not include this instance
            self.one_batch.append(self._instances[self._idx_to_read_next_batch])
            self.word_num_accum_so_far += sz
            self.max_len = max(sz, self.max_len)
            self._idx_to_read_next_batch += 1
            if self._idx_to_read_next_batch >= self.size() or \
                    (self._word_num_one_batch <= 0 and len(self.one_batch) == self._sent_num_one_batch):
                break

        return self.one_batch, self.word_num_accum_so_far, self.max_len
