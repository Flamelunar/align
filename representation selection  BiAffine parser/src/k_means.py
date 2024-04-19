#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Timothy Dozat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import sys

import numpy as np


# ***************************************************************
class KMeans(object):
    """"""

    # =============================================================
    def __init__(self, k, len_cntr):
        """"""

        # Error checking
        if len(len_cntr) < k:
            raise ValueError('Trying to sort %d data points into %d buckets' % (len(len_cntr), k))

        # Initialize variables
        self._k = k
        self._len_cntr = len_cntr
        self._uniq_lengths = sorted(self._len_cntr.keys())
        self._max_len_in_buckets = []  # zhenghua: split point (len)
        self._split2len_idx = {}  # zhenghua: split point to len-idx (in self._uniq_lengths, ascending order)
        self._len2bucket_idx = {}  # length to split_idx (in self._max_len_in_buckets)
        self._split_cntr = Counter()  # split point to data counter

        # print(self._uniq_lengths, flush=True)
        # Initialize the splits evenly
        lengths = []
        for length, count in self._len_cntr.items():
            lengths.extend([length] * count)
        lengths.sort()
        #print(lengths)
        self._max_len_in_buckets = [np.max(split) for split in np.array_split(lengths, self._k)]
        # print(self._max_len_in_buckets, flush=True)

        # zhenghua: 我目前还不太理解这个算法，是不是一次调整就可以到位
        # 第一次，调整split points
        # 第二次，如果发现，还能调整，那说明之前的代码有问题，报错
        # 可以考虑把2变大
        for try_time in range(2):
            change_flag = False
            i = len(self._max_len_in_buckets) - 1
            while i > 0:
                # add the first condition by zhenghua
                while self._max_len_in_buckets[i - 1] > self._uniq_lengths[0] \
                        and (self._max_len_in_buckets[i - 1] >= self._max_len_in_buckets[i]
                             or self._max_len_in_buckets[i - 1] not in self._len_cntr):
                    self._max_len_in_buckets[i - 1] -= 1
                    change_flag = True
                i -= 1

            i = 1
            while i < len(self._max_len_in_buckets) - 1:
                # add the first condition by zhenghua
                while self._max_len_in_buckets[i] < self._uniq_lengths[-1] \
                        and (self._max_len_in_buckets[i] <= self._max_len_in_buckets[i - 1]
                             or self._max_len_in_buckets[i] not in self._len_cntr):
                    self._max_len_in_buckets[i] += 1
                    change_flag = True
                i += 1
            if not change_flag:
                break
        else:
            raise Exception('may be the bucket num is too many, initialization cannot be done')

        # Reindex everything
        split_idx = 0
        split = self._max_len_in_buckets[split_idx]
        for len_idx, length in enumerate(self._uniq_lengths):
            count = self._len_cntr[length]
            self._split_cntr[split] += count
            if length == split:
                self._split2len_idx[split] = len_idx
                split_idx += 1
                if split_idx < len(self._max_len_in_buckets):
                    split = self._max_len_in_buckets[split_idx]
                    self._split_cntr[split] = 0
            elif length > split:
                raise IndexError()
        # print('iterate', flush=True)

        # Iterate
        old_splits = None
        record_all_splits_after_some_time = []
        # print('0) Initial splits: %s; Initial mass: %d' % (self._max_len_in_buckets, self.get_mass()))
        i = 0
        while self._max_len_in_buckets != old_splits:
            old_splits = list(self._max_len_in_buckets)
            if i >= 0:
                if old_splits in record_all_splits_after_some_time:
                    # print('haha: recycle %d' % i, flush=True)
                    break
                record_all_splits_after_some_time.append(list(old_splits))
                # print(i, ':', old_splits, flush=True)
            # print('before recenter', flush=True)
            self.recenter()
            # print('after recenter', flush=True)
            i += 1
        # print('%d) Final splits: %s; Final mass: %d' % (i, self._max_len_in_buckets, self.get_mass()))

        self.reindex()
        # print('k means returned', flush=True)
        # print(self._uniq_lengths)
        return

    # =============================================================
    def recenter(self):
        for split_idx in range(len(self._max_len_in_buckets)):
            split = self._max_len_in_buckets[split_idx]
            len_idx = self._split2len_idx[split]
            if split == self._max_len_in_buckets[-1]:
                continue
            right_split = self._max_len_in_buckets[split_idx + 1]

            # Try shifting the centroid to the left
            if len_idx > 0 and self._uniq_lengths[len_idx - 1] not in self._split_cntr:
                new_split = self._uniq_lengths[len_idx - 1]
                left_delta = self._len_cntr[split] * (right_split - new_split) - self._split_cntr[split] * (
                            split - new_split)
                if left_delta < 0:
                    self._max_len_in_buckets[split_idx] = new_split
                    self._split2len_idx[new_split] = len_idx - 1
                    del self._split2len_idx[split]
                    self._split_cntr[split] -= self._len_cntr[split]
                    self._split_cntr[right_split] += self._len_cntr[split]
                    self._split_cntr[new_split] = self._split_cntr[split]
                    del self._split_cntr[split]

            # Try shifting the centroid to the right
            elif len_idx < len(self._uniq_lengths) - 2 and self._uniq_lengths[len_idx + 1] not in self._split_cntr:
                new_split = self._uniq_lengths[len_idx + 1]
                right_delta = self._split_cntr[split] * (new_split - split) - self._len_cntr[split] * (
                            new_split - split)
                if right_delta <= 0:
                    self._max_len_in_buckets[split_idx] = new_split
                    self._split2len_idx[new_split] = len_idx + 1
                    del self._split2len_idx[split]
                    self._split_cntr[split] += self._len_cntr[split]
                    self._split_cntr[right_split] -= self._len_cntr[split]
                    self._split_cntr[new_split] = self._split_cntr[split]
                    del self._split_cntr[split]
        return

        # =============================================================

    def get_mass(self):
        """"""

        mass = 0
        split_idx = 0
        split = self._max_len_in_buckets[split_idx]
        for len_idx, length in enumerate(self._uniq_lengths):
            count = self._len_cntr[length]
            mass += split * count
            if length == split:
                split_idx += 1
                if split_idx < len(self._max_len_in_buckets):
                    split = self._max_len_in_buckets[split_idx]
        return mass

    # =============================================================
    def reindex(self):
        """"""
        self._len2bucket_idx = {}
        last_split = -1
        for split_idx, split in enumerate(self._max_len_in_buckets):
            self._len2bucket_idx.update(
                dict(zip(range(last_split+1, split+1), [split_idx] * (split - (last_split)))))
            last_split = split  # add this line by zhenghua (a bug)
        return

        # =============================================================

    def __len__(self):
        return self._k

    def __iter__(self):
        return (split for split in self.splits)

    def __getitem__(self, key):
        return self._max_len_in_buckets[key]

    # =============================================================
    @property
    def max_len_in_buckets(self):
        return self._max_len_in_buckets

    @property
    def len2bucket_idx(self):
        return self._len2bucket_idx


# ***************************************************************
if __name__ == '__main__':
    np.random.seed(123)
    len_cntr = Counter()
    for i in range(50):
        # len_cntr[1+int(10**(1+np.random.randn()))] += 1
        len_cntr[1 + np.random.randint(1, 10)] += 1
    print(len_cntr)
    kmeans = KMeans(2, len_cntr)
    print('splits: ', kmeans.max_len_in_buckets)
    print('len2split_idx: ', kmeans.len2bucket_idx)
